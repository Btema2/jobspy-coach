import html
from unittest.mock import MagicMock
import httpx
import pytest

from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.templates import format_digest_chunks, format_wow_alert
from src.storage.models import JobPosting


def test_format_wow_alert_escapes_html_and_builds_markup():
    job = JobPosting(
        dedup_hash="hash1",
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/123",
        title="Executive Coach <Global & Poland>",
        company="Talent & Leadership Sp. z o.o.",
        location="Warsaw & Remote",
        description="Job desc",
        eval_score=9.5,
        eval_is_wow=True,
        eval_summary_uk="Чудова роль для роботи з топами компанії.",
        eval_reasoning="Top match",
        eval_languages=["English", "Polish"],
        eval_pros=["Great salary"],
        eval_cons=[],
    )
    html_text, markup = format_wow_alert(job)
    assert "&lt;Global &amp; Poland&gt;" in html_text
    assert "Talent &amp; Leadership" in html_text
    assert "9.5 / 10" in html_text
    assert "Чудова роль для роботи з топами компанії." in html_text
    assert markup["inline_keyboard"][0][0]["url"] == job.job_url


def test_format_digest_chunks_split_long_messages():
    jobs = []
    for i in range(25):
        jobs.append(
            JobPosting(
                dedup_hash=f"hash_{i}",
                site="indeed",
                job_url=f"https://example.com/job/{i}",
                title=f"Coaching Role {i} with long title and details",
                company=f"Company {i}",
                location="Remote Poland",
                description="Long description",
                eval_score=8.0 - (i * 0.1),
                eval_summary_uk="Опис ролі та ключові обов'язки тренера або коуча команди.",
            )
        )

    chunks = format_digest_chunks(jobs, date_str="15.10.2026")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) <= 3800
        assert "Ранковий Дайджест Вакансій Коучингу" in chunk or "Частина" in chunk


def test_telegram_notifier_send_wow_alert_success(monkeypatch):
    mock_post = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    monkeypatch.setattr("httpx.Client.post", mock_post)

    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    job = JobPosting(
        dedup_hash="hash1",
        site="linkedin",
        job_url="https://example.com",
        title="Senior Coach",
        company="Corp",
        location="Remote",
        description="Desc",
        eval_score=9.5,
        eval_is_wow=True,
        eval_summary_uk="Чудова можливість.",
    )

    success = notifier.send_wow_alert(job)
    assert success is True
    assert mock_post.called

    call_args, call_kwargs = mock_post.call_args
    assert "https://api.telegram.org/botfake-token/sendMessage" in call_args
    payload = call_kwargs["json"]
    assert payload["chat_id"] == "12345"
    assert "Senior Coach" in payload["text"]
    assert payload["parse_mode"] == "HTML"
    assert "reply_markup" in payload


def test_telegram_notifier_rate_limit_and_error_handling(monkeypatch):
    # 1. Missing credentials
    notifier_no_token = TelegramNotifier(token="", chat_id="12345")
    assert notifier_no_token.send_message("test") is False

    notifier_no_chat = TelegramNotifier(token="fake-token", chat_id="")
    assert notifier_no_chat.send_message("test") is False

    # 2. HTTP error (e.g. 429 Too Many Requests)
    mock_post = MagicMock()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    req = httpx.Request("POST", "https://api.telegram.org")
    mock_resp_429.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Rate limit exceeded", request=req, response=mock_resp_429
    )
    mock_post.return_value = mock_resp_429
    monkeypatch.setattr("httpx.Client.post", mock_post)

    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    assert notifier.send_message("test rate limit") is False

    # 3. Connection timeout error
    mock_post.side_effect = httpx.ConnectTimeout("Connection timed out")
    assert notifier.send_message("test timeout") is False

    # 4. send_daily_digest empty jobs returns 0
    assert notifier.send_daily_digest([], "15.10.2026") == 0

    # 5. send_daily_digest rate limit pacing (1 msg/sec sleep)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)
    # mock send_message to return True
    send_message_mock = MagicMock(return_value=True)
    monkeypatch.setattr(notifier, "send_message", send_message_mock)

    # Create enough jobs to produce >1 chunk
    jobs = [
        JobPosting(
            dedup_hash=f"hash_{i}",
            site="indeed",
            job_url=f"https://example.com/job/{i}",
            title=f"Coaching Role {i} with long description to force chunking",
            company=f"Company {i}",
            location="Remote Poland",
            description="Long description",
            eval_score=8.0,
            eval_summary_uk="Опис ролі та обов'язків для дайджесту вакансій.",
        )
        for i in range(25)
    ]

    sent_count = notifier.send_daily_digest(jobs, "15.10.2026")
    assert sent_count > 1
    assert mock_sleep.call_count == sent_count
    mock_sleep.assert_called_with(1.0)


def test_format_wow_alert_with_salary_and_fallback_fields():
    job = JobPosting(
        dedup_hash="hash_salary",
        site="pracuj_pl",
        job_url="https://pracuj.pl/job/1",
        title="Agile Coach",
        company="FinTech Ltd",
        location="Krakow",
        description="Job desc",
        salary_min=18000.0,
        salary_max=24000.0,
        salary_currency="PLN",
        eval_score=None,
        eval_summary_uk=None,
        eval_reasoning=None,
        eval_languages=[],
    )
    html_text, markup = format_wow_alert(job)
    assert "💰 <b>Вилка:</b> 18000.0 - 24000.0 PLN" in html_text
    assert "⭐️ <b>Оцінка: 9.0 / 10</b>" in html_text
    assert "Опис відсутній." in html_text
    assert "Високий рівень відповідності профілю." in html_text
    assert "🌐 <b>Мови:</b>" not in html_text


def test_format_digest_chunks_empty():
    chunks = format_digest_chunks([], date_str="16.10.2026")
    assert len(chunks) == 1
    assert "Знайдено нових актуальних позицій: <b>0</b>" in chunks[0]


def test_telegram_notifier_from_settings():
    from src.config import Settings

    custom_settings = Settings(
        TELEGRAM_BOT_TOKEN="my-bot-token",
        TELEGRAM_CHAT_ID="999888",
    )
    notifier = TelegramNotifier.from_settings(custom_settings, timeout=20.0)
    assert notifier.token == "my-bot-token"
    assert notifier.chat_id == "999888"
    assert notifier.timeout == 20.0

    # Default settings fallback
    default_notifier = TelegramNotifier.from_settings()
    assert isinstance(default_notifier, TelegramNotifier)


