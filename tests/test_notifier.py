import html
from unittest.mock import MagicMock
import httpx
import pytest

from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.templates import (
    format_digest_chunks,
    format_version_update_notice,
    format_wow_alert,
)
from src.scraper.version_checker import VersionCheckResult
from src.storage.models import JobPosting


def test_format_wow_alert_escapes_html_and_builds_markup():
    job = JobPosting(
        dedup_hash="hash1",
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/123?ref=test&src=jobspy",
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
    assert markup["inline_keyboard"][0][0]["text"] == "🚀 Apply / View Job"


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


def test_format_digest_chunks_escapes_job_url_entities():
    job = JobPosting(
        dedup_hash="hash_url",
        site="linkedin",
        job_url="https://www.linkedin.com/jobs/view/123456?refId=abc&trackingId=def==&position=1",
        title="Career Coach & Mentor",
        company="Coaching Co. <PL>",
        location="Warsaw",
        description="Desc",
        eval_score=8.5,
        eval_summary_uk="Тестовий опис ролі.",
    )
    chunks = format_digest_chunks([job], date_str="16.10.2026 & Autumn")
    assert len(chunks) == 1
    chunk = chunks[0]
    expected_url = "https://www.linkedin.com/jobs/view/123456?refId=abc&amp;trackingId=def==&amp;position=1"
    assert f'<a href="{expected_url}">' in chunk
    assert "16.10.2026 &amp; Autumn" in chunk


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
    assert payload["link_preview_options"] == {"is_disabled": True}
    assert "disable_web_page_preview" not in payload
    assert "reply_markup" in payload
    notifier.close()


def test_telegram_notifier_rate_limit_and_error_handling(monkeypatch):
    # 1. Missing credentials
    notifier_no_token = TelegramNotifier(token="", chat_id="12345")
    assert notifier_no_token.send_message("test") is False
    notifier_no_token.close()

    notifier_no_chat = TelegramNotifier(token="fake-token", chat_id="")
    assert notifier_no_chat.send_message("test") is False
    notifier_no_chat.close()

    # 2. Connection timeout error
    mock_post = MagicMock(side_effect=httpx.ConnectTimeout("Connection timed out"))
    monkeypatch.setattr("httpx.Client.post", mock_post)

    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    assert notifier.send_message("test timeout") is False
    notifier.close()

    # 3. HTTP 500 error (non-429)
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    req = httpx.Request("POST", "https://api.telegram.org")
    mock_resp_500.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=req, response=mock_resp_500
    )
    mock_post = MagicMock(return_value=mock_resp_500)
    monkeypatch.setattr("httpx.Client.post", mock_post)

    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    assert notifier.send_message("test server error") is False
    notifier.close()

    # 4. send_daily_digest empty jobs returns 0
    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    assert notifier.send_daily_digest([], "15.10.2026") == 0
    notifier.close()

    # 5. send_daily_digest rate limit pacing (1 msg/sec sleep)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)
    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    send_message_mock = MagicMock(return_value=True)
    monkeypatch.setattr(notifier, "send_message", send_message_mock)

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
    notifier.close()


def test_telegram_notifier_429_retry_success(monkeypatch):
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.json.return_value = {"ok": False, "parameters": {"retry_after": 3}}
    resp_429.headers = {}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"ok": True}
    resp_200.raise_for_status.return_value = None

    mock_post = MagicMock(side_effect=[resp_429, resp_200])
    monkeypatch.setattr("httpx.Client.post", mock_post)

    with TelegramNotifier(token="fake-token", chat_id="12345") as notifier:
        result = notifier.send_message("test 429 retry")
        assert result is True
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(3.0)


def test_telegram_notifier_429_retry_exhausted(monkeypatch):
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    resp_429_invalid_header = MagicMock()
    resp_429_invalid_header.status_code = 429
    resp_429_invalid_header.json.side_effect = Exception("Malformed JSON")
    resp_429_invalid_header.headers = {"Retry-After": "not-an-integer"}

    resp_429_valid_header = MagicMock()
    resp_429_valid_header.status_code = 429
    resp_429_valid_header.json.return_value = {"ok": False}
    resp_429_valid_header.headers = {"Retry-After": "4"}

    mock_post = MagicMock(side_effect=[resp_429_invalid_header, resp_429_valid_header, resp_429_valid_header])
    monkeypatch.setattr("httpx.Client.post", mock_post)

    with TelegramNotifier(token="fake-token", chat_id="12345") as notifier:
        result = notifier.send_message("test 429 exhaust")
        assert result is False
        # Initial attempt + 2 retries = 3 calls
        assert mock_post.call_count == 3
        # First retry defaulted to 5.0, second retry used header 4.0
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5.0)
        mock_sleep.assert_any_call(4.0)



def test_telegram_notifier_context_manager_and_client_lifecycle():
    # Managed client closed on exit
    with TelegramNotifier(token="fake-token", chat_id="12345") as notifier:
        assert not notifier._client.is_closed
    assert notifier._client.is_closed

    # External client not closed by TelegramNotifier
    external_client = httpx.Client()
    try:
        with TelegramNotifier(token="fake-token", chat_id="12345", client=external_client) as notifier:
            assert notifier._client is external_client
        assert not external_client.is_closed
    finally:
        external_client.close()


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
    assert "💰 <b>Вилка:</b> 18000 - 24000 PLN" in html_text
    assert "⭐️ <b>Оцінка: 9.0 / 10</b>" in html_text
    assert "Опис відсутній." in html_text
    assert "Високий рівень відповідності профілю." in html_text
    assert "🌐 <b>Мови:</b>" not in html_text


def test_format_wow_alert_single_sided_salaries():
    # Min only
    job_min = JobPosting(
        dedup_hash="hash_min",
        site="linkedin",
        job_url="https://linkedin.com/1",
        title="Lead Coach",
        company="Co",
        location="Remote",
        description="Desc",
        salary_min=25000.0,
        salary_currency="PLN",
    )
    html_min, _ = format_wow_alert(job_min)
    assert "💰 <b>Вилка:</b> від 25000 PLN" in html_min
    assert " - " not in html_min

    # Max only
    job_max = JobPosting(
        dedup_hash="hash_max",
        site="indeed",
        job_url="https://indeed.com/2",
        title="Senior Coach",
        company="Co",
        location="Remote",
        description="Desc",
        salary_max=35000.0,
        salary_currency="PLN",
    )
    html_max, _ = format_wow_alert(job_max)
    assert "💰 <b>Вилка:</b> до 35000 PLN" in html_max
    assert " - " not in html_max


def test_format_wow_alert_defensive_length_guard():
    job_long = JobPosting(
        dedup_hash="hash_long",
        site="linkedin",
        job_url="https://linkedin.com/long",
        title="Executive Coach",
        company="Global Leadership",
        location="Remote",
        description="Desc",
        eval_score=9.5,
        eval_is_wow=True,
        eval_summary_uk="A" * 1000,
        eval_reasoning="B" * 1200,
    )
    html_text, _ = format_wow_alert(job_long)
    # The total message must stay well below 4096 chars
    assert len(html_text) <= 2500
    assert "A" * 650 not in html_text
    assert "B" * 850 not in html_text
    assert "..." in html_text


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
    notifier.close()

    # Default settings fallback
    default_notifier = TelegramNotifier.from_settings()
    assert isinstance(default_notifier, TelegramNotifier)
    default_notifier.close()


def test_format_version_update_notice():
    result = VersionCheckResult(
        has_update=True,
        installed_version="1.3.0",
        latest_version="1.4.0",
    )
    notice = format_version_update_notice(result)
    assert "ℹ️" in notice
    assert "Доступне оновлення python-jobspy: v1.4.0" in notice
    assert "(встановлено: v1.3.0)" in notice
    assert "<code>pip install -U python-jobspy</code>" in notice

    # Test HTML escaping of version strings
    unsafe_result = VersionCheckResult(
        has_update=True,
        installed_version="1.3.0<beta>",
        latest_version="1.4.0&test",
    )
    unsafe_notice = format_version_update_notice(unsafe_result)
    assert "v1.4.0&amp;test" in unsafe_notice
    assert "v1.3.0&lt;beta&gt;" in unsafe_notice
    assert "<beta>" not in unsafe_notice


def test_format_digest_chunks_with_update_notice():
    job = JobPosting(
        dedup_hash="hash_update_notice",
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/999",
        title="Agile Coach",
        company="Coaching Org",
        location="Kyiv",
        description="Desc",
        eval_score=8.5,
        eval_summary_uk="Опис позиції коуча.",
    )
    notice = "\n\nℹ️ <i>Доступне оновлення python-jobspy: v1.4.0 (встановлено: v1.3.0).\nЩоб оновити: <code>pip install -U python-jobspy</code></i>"

    # 1. Single job with notice
    chunks = format_digest_chunks([job], date_str="22.10.2026", update_notice=notice)
    assert len(chunks) == 1
    assert notice in chunks[0]
    assert len(chunks[0]) <= 3800

    # 2. Empty jobs with notice
    empty_chunks = format_digest_chunks([], date_str="22.10.2026", update_notice=notice)
    assert len(empty_chunks) == 1
    assert notice in empty_chunks[0]
    assert "Знайдено нових актуальних позицій: <b>0</b>" in empty_chunks[0]
    assert len(empty_chunks[0]) <= 3800

    # 3. Multi-chunk split: notice must be appended ONLY to final chunk
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
    multi_chunks = format_digest_chunks(jobs, date_str="22.10.2026", update_notice=notice)
    assert len(multi_chunks) > 1
    for chunk in multi_chunks:
        assert len(chunk) <= 3800
    assert notice in multi_chunks[-1]
    for chunk in multi_chunks[:-1]:
        assert notice not in chunk


def test_format_digest_chunks_without_update_notice():
    job = JobPosting(
        dedup_hash="hash_no_notice",
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/888",
        title="Career Coach",
        company="Talent Inc",
        location="Warsaw",
        description="Desc",
        eval_score=8.0,
        eval_summary_uk="Тестовий опис ролі.",
    )
    chunks_default = format_digest_chunks([job], date_str="22.10.2026")
    chunks_none = format_digest_chunks([job], date_str="22.10.2026", update_notice=None)
    assert chunks_default == chunks_none
    assert "Доступне оновлення" not in chunks_default[0]


def test_send_daily_digest_forwards_update_notice(monkeypatch):
    mock_chunks = ["chunk1", "chunk2"]
    mock_format = MagicMock(return_value=mock_chunks)
    monkeypatch.setattr("src.notifier.telegram_notifier.format_digest_chunks", mock_format)

    mock_send = MagicMock(return_value=True)
    notifier = TelegramNotifier(token="fake-token", chat_id="12345")
    monkeypatch.setattr(notifier, "send_message", mock_send)
    monkeypatch.setattr("time.sleep", MagicMock())

    job = JobPosting(
        dedup_hash="hash_fwd",
        site="linkedin",
        job_url="https://linkedin.com/1",
        title="Coach",
        company="Co",
        location="Remote",
        description="Desc",
    )
    notice = "sample update notice"
    count = notifier.send_daily_digest([job], date_str="22.10.2026", update_notice=notice)

    assert count == 2
    mock_format.assert_called_once_with([job], date_str="22.10.2026", update_notice=notice)
    notifier.close()

