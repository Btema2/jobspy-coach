import html
from typing import Any, Dict, List, Tuple
from src.storage.models import JobPosting


def format_wow_alert(job: JobPosting) -> Tuple[str, Dict[str, Any]]:
    title_esc = html.escape(job.title or "")
    comp_esc = html.escape(job.company or "")
    loc_esc = html.escape(job.location or "")
    summary_esc = html.escape(job.eval_summary_uk or "Опис відсутній.")
    reasoning_esc = html.escape(job.eval_reasoning or "Високий рівень відповідності профілю.")
    score_str = f"{job.eval_score:.1f}" if job.eval_score is not None else "9.0"

    salary_line = ""
    if job.salary_min or job.salary_max:
        curr = html.escape(job.salary_currency or "")
        sal = f"{job.salary_min or ''} - {job.salary_max or ''} {curr}".strip()
        salary_line = f"\n💰 <b>Вилка:</b> {sal}"

    langs_line = ""
    if job.eval_languages:
        langs_esc = html.escape(", ".join(job.eval_languages))
        langs_line = f"\n🌐 <b>Мови:</b> {langs_esc}"

    msg = (
        f"🚨 <b>WOW JOB ALERT | Топ-Вакансія Коучингу</b>\n"
        f"⭐️ <b>Оцінка: {score_str} / 10</b>\n\n"
        f"💼 <b>Посада:</b> {title_esc}\n"
        f"🏢 <b>Компанія:</b> {comp_esc}\n"
        f"📍 <b>Локація:</b> {loc_esc}{salary_line}{langs_line}\n\n"
        f"💡 <b>Про вакансію:</b>\n{summary_esc}\n\n"
        f"🎯 <b>Чому це WOW:</b>\n{reasoning_esc}"
    )

    markup = {
        "inline_keyboard": [
            [{"text": "🚀 Відкрити вакансію", "url": job.job_url}]
        ]
    }
    return msg, markup


def format_digest_chunks(jobs: List[JobPosting], date_str: str) -> List[str]:
    header = (
        f"☕️ <b>Ранковий Дайджест Вакансій Коучингу | {date_str}</b>\n"
        f"Знайдено нових актуальних позицій: <b>{len(jobs)}</b>\n\n"
    )
    footer = "\n<i>Всі позиції збережено в базі. Бажаємо продуктивного дня!</i>"

    chunks = []
    current_chunk = header
    chunk_index = 1

    for idx, job in enumerate(jobs, start=1):
        score_str = f"{job.eval_score:.1f}" if job.eval_score is not None else "-"
        title_esc = html.escape(job.title or "")
        comp_esc = html.escape(job.company or "")
        loc_esc = html.escape(job.location or "")
        summary_esc = html.escape(job.eval_summary_uk or "")

        item_str = (
            f"{idx}. ⭐️ <b>{score_str}/10</b> — <b>{title_esc}</b>\n"
            f"   🏢 {comp_esc} | 📍 {loc_esc}\n"
            f"   💡 {summary_esc}\n"
            f"   🔗 <a href=\"{job.job_url}\">Переглянути вакансію</a>\n\n"
        )

        if len(current_chunk) + len(item_str) + len(footer) > 3800:
            current_chunk += footer
            chunks.append(current_chunk)
            chunk_index += 1
            current_chunk = f"☕️ <b>Ранковий Дайджест | Частина {chunk_index}</b>\n\n" + item_str
        else:
            current_chunk += item_str

    current_chunk += footer
    chunks.append(current_chunk)
    return chunks
