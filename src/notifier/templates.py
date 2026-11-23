import html
from typing import Any, Dict, List, Optional, Tuple
from src.i18n import t
from src.scraper.version_checker import VersionCheckResult
from src.storage.models import JobPosting


def _format_salary_val(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except (ValueError, TypeError):
        return str(val)


def _format_salary_line(job: JobPosting, lang: Optional[str] = None) -> str:
    min_val = job.salary_min if (job.salary_min is not None and job.salary_min > 0) else None
    max_val = job.salary_max if (job.salary_max is not None and job.salary_max > 0) else None

    if min_val is None and max_val is None:
        return ""

    min_str = _format_salary_val(min_val)
    max_str = _format_salary_val(max_val)

    curr = f" {html.escape(job.salary_currency.strip())}" if job.salary_currency and job.salary_currency.strip() else ""

    if min_str is not None and max_str is not None:
        sal = f"{min_str} - {max_str}{curr}"
    elif min_str is not None:
        sal = t("salary_from", lang, amount=f"{min_str}{curr}")
    elif max_str is not None:
        sal = t("salary_to", lang, amount=f"{max_str}{curr}")
    else:
        return ""

    return t("salary_line", lang, salary=sal)


def _truncate_escaped_html(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    cutoff = max_length - 3
    truncated = text[:cutoff]
    last_amp = truncated.rfind("&")
    last_semi = truncated.rfind(";")
    if last_amp != -1 and last_amp > last_semi:
        truncated = truncated[:last_amp]
    return truncated.rstrip() + "..."


def format_wow_alert(job: JobPosting, lang: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    title_esc = html.escape(job.title or "")
    comp_esc = html.escape(job.company or "")
    loc_esc = html.escape(job.location or "")

    summary_raw = job.eval_summary_uk or t("wow_summary_missing", lang)
    summary_esc = html.escape(summary_raw)
    if len(summary_esc) > 600:
        summary_esc = _truncate_escaped_html(summary_esc, 600)

    reasoning_raw = job.eval_reasoning or t("wow_reasoning_missing", lang)
    reasoning_esc = html.escape(reasoning_raw)
    if len(reasoning_esc) > 800:
        reasoning_esc = _truncate_escaped_html(reasoning_esc, 800)

    score_str = f"{job.eval_score:.1f}" if job.eval_score is not None else "9.0"

    salary_line = _format_salary_line(job, lang)

    langs_line = ""
    if job.eval_languages:
        langs_esc = html.escape(", ".join(job.eval_languages))
        langs_line = t("wow_languages_line", lang, languages=langs_esc)

    msg = t(
        "wow_body",
        lang,
        score=score_str,
        title=title_esc,
        company=comp_esc,
        location=loc_esc,
        salary_line=salary_line,
        languages_line=langs_line,
        summary=summary_esc,
        reasoning=reasoning_esc,
    )

    markup = {
        "inline_keyboard": [
            [{"text": t("wow_button", lang), "url": job.job_url}]
        ]
    }
    return msg, markup


def format_version_update_notice(result: VersionCheckResult, lang: Optional[str] = None) -> str:
    """Formats the update notice for the Telegram HTML digest (language table per the profile)."""
    latest_esc = html.escape(result.latest_version)
    installed_esc = html.escape(result.installed_version)
    return t("update_notice", lang, latest=latest_esc, installed=installed_esc)


def format_digest_chunks(
    jobs: List[JobPosting],
    date_str: str,
    update_notice: Optional[str] = None,
    lang: Optional[str] = None,
) -> List[str]:
    date_esc = html.escape(date_str or "")
    header = t("digest_header", lang, date=date_esc, count=str(len(jobs)))
    standard_footer = t("digest_footer", lang)
    final_footer = standard_footer + (update_notice if update_notice else "")

    chunks = []
    current_chunk = header
    chunk_index = 1
    total_jobs = len(jobs)

    for idx, job in enumerate(jobs, start=1):
        score_str = f"{job.eval_score:.1f}" if job.eval_score is not None else "-"
        title_esc = html.escape(job.title or "")
        comp_esc = html.escape(job.company or "")
        loc_esc = html.escape(job.location or "")
        summary_esc = html.escape(job.eval_summary_uk or "")
        url_esc = html.escape(job.job_url or "", quote=True)

        item_str = t(
            "digest_item",
            lang,
            idx=str(idx),
            score=score_str,
            title=title_esc,
            company=comp_esc,
            location=loc_esc,
            summary=summary_esc,
            url=url_esc,
        )

        needed_footer = final_footer if idx == total_jobs else standard_footer
        if len(current_chunk) + len(item_str) + len(needed_footer) > 3800:
            current_chunk += standard_footer
            chunks.append(current_chunk)
            chunk_index += 1
            current_chunk = t("digest_part_header", lang, part=str(chunk_index)) + item_str
        else:
            current_chunk += item_str

    current_chunk += final_footer
    chunks.append(current_chunk)
    return chunks
