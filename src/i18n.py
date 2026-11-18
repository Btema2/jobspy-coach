"""Static message tables for alerts and digests.

``"uk"`` holds the strings that were hard-coded before the profile had an ``output_language`` and is what an
unset language keeps using (byte-identical output for imported installs); ``"en"`` is the neutral default for
every other language without a table. Values are ``str.format`` templates over HTML; every key lists the
placeholders it must contain (``REQUIRED_PLACEHOLDERS``), checked by the tests.
"""
from typing import Dict, FrozenSet, Optional

DEFAULT_LANGUAGE = "en"
LEGACY_LANGUAGE = "uk"

_UK: Dict[str, str] = {
    "salary_from": "від {amount}",
    "salary_to": "до {amount}",
    "salary_line": "\n💰 <b>Вилка:</b> {salary}",
    "wow_summary_missing": "Опис відсутній.",
    "wow_reasoning_missing": "Високий рівень відповідності профілю.",
    "wow_languages_line": "\n🌐 <b>Мови:</b> {languages}",
    "wow_body": (
        "🚨 <b>WOW JOB ALERT | Топ-Вакансія Коучингу</b>\n"
        "⭐️ <b>Оцінка: {score} / 10</b>\n\n"
        "💼 <b>Посада:</b> {title}\n"
        "🏢 <b>Компанія:</b> {company}\n"
        "📍 <b>Локація:</b> {location}{salary_line}{languages_line}\n\n"
        "💡 <b>Про вакансію:</b>\n{summary}\n\n"
        "🎯 <b>Чому це WOW:</b>\n{reasoning}"
    ),
    "wow_button": "🚀 Apply / View Job",
    "update_notice": (
        "\n\nℹ️ <i>Доступне оновлення python-jobspy: v{latest} (встановлено: v{installed}).\n"
        "Щоб оновити: <code>pip install -U python-jobspy</code></i>"
    ),
    "digest_header": (
        "☕️ <b>Ранковий Дайджест Вакансій Коучингу | {date}</b>\n"
        "Знайдено нових актуальних позицій: <b>{count}</b>\n\n"
    ),
    "digest_footer": "\n<i>Всі позиції збережено в базі. Бажаємо продуктивного дня!</i>",
    "digest_item": (
        "{idx}. ⭐️ <b>{score}/10</b> — <b>{title}</b>\n"
        "   🏢 {company} | 📍 {location}\n"
        "   💡 {summary}\n"
        "   🔗 <a href=\"{url}\">Переглянути вакансію</a>\n\n"
    ),
    "digest_part_header": "☕️ <b>Ранковий Дайджест | Частина {part}</b>\n\n",
    "test_telegram": "👋 <b>JobSpy Coach Daemon</b> | Підключення успішне! Тестове сповіщення доставлено.",
}

_EN: Dict[str, str] = {
    "salary_from": "from {amount}",
    "salary_to": "up to {amount}",
    "salary_line": "\n💰 <b>Salary:</b> {salary}",
    "wow_summary_missing": "No description available.",
    "wow_reasoning_missing": "Strong match with the profile.",
    "wow_languages_line": "\n🌐 <b>Languages:</b> {languages}",
    "wow_body": (
        "🚨 <b>WOW JOB ALERT | Top Match</b>\n"
        "⭐️ <b>Score: {score} / 10</b>\n\n"
        "💼 <b>Title:</b> {title}\n"
        "🏢 <b>Company:</b> {company}\n"
        "📍 <b>Location:</b> {location}{salary_line}{languages_line}\n\n"
        "💡 <b>About the job:</b>\n{summary}\n\n"
        "🎯 <b>Why it is a WOW:</b>\n{reasoning}"
    ),
    "wow_button": "🚀 Apply / View Job",
    "update_notice": (
        "\n\nℹ️ <i>python-jobspy update available: v{latest} (installed: v{installed}).\n"
        "To update: <code>pip install -U python-jobspy</code></i>"
    ),
    "digest_header": (
        "☕️ <b>Morning Job Digest | {date}</b>\n"
        "New relevant positions found: <b>{count}</b>\n\n"
    ),
    "digest_footer": "\n<i>All positions are saved in the database. Have a productive day!</i>",
    "digest_item": (
        "{idx}. ⭐️ <b>{score}/10</b> — <b>{title}</b>\n"
        "   🏢 {company} | 📍 {location}\n"
        "   💡 {summary}\n"
        "   🔗 <a href=\"{url}\">View job</a>\n\n"
    ),
    "digest_part_header": "☕️ <b>Morning Digest | Part {part}</b>\n\n",
    "test_telegram": "👋 <b>JobSpy Coach Daemon</b> | Connection successful! Test notification delivered.",
}

TABLES: Dict[str, Dict[str, str]] = {"en": _EN, "uk": _UK}

REQUIRED_PLACEHOLDERS: Dict[str, FrozenSet[str]] = {
    "salary_from": frozenset({"amount"}),
    "salary_to": frozenset({"amount"}),
    "salary_line": frozenset({"salary"}),
    "wow_summary_missing": frozenset(),
    "wow_reasoning_missing": frozenset(),
    "wow_languages_line": frozenset({"languages"}),
    "wow_body": frozenset(
        {"score", "title", "company", "location", "salary_line", "languages_line", "summary", "reasoning"}
    ),
    "wow_button": frozenset(),
    "update_notice": frozenset({"latest", "installed"}),
    "digest_header": frozenset({"date", "count"}),
    "digest_footer": frozenset(),
    "digest_item": frozenset({"idx", "score", "title", "company", "location", "summary", "url"}),
    "digest_part_header": frozenset({"part"}),
    "test_telegram": frozenset(),
}


def table_language(output_language: Optional[str]) -> str:
    """The table used for a profile language: unset keeps the legacy strings, unknown codes fall back to English."""
    if output_language is None:
        return LEGACY_LANGUAGE
    return output_language if output_language in TABLES else DEFAULT_LANGUAGE


def t(key: str, output_language: Optional[str], **values: str) -> str:
    return TABLES[table_language(output_language)][key].format(**values)
