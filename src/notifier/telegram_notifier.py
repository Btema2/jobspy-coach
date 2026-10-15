import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from src.notifier.templates import format_digest_chunks, format_wow_alert
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: float = 15.0):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram Bot Token or Chat ID not configured. Skipping notification.")
            return False

        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.api_url, json=payload)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def send_wow_alert(self, job: JobPosting) -> bool:
        text, markup = format_wow_alert(job)
        return self.send_message(text, reply_markup=markup)

    def send_daily_digest(self, jobs: List[JobPosting], date_str: str) -> int:
        if not jobs:
            return 0

        chunks = format_digest_chunks(jobs, date_str=date_str)
        sent_count = 0
        for chunk in chunks:
            if self.send_message(chunk):
                sent_count += 1
            # Respect 1 msg/sec pacing
            time.sleep(1.0)
        return sent_count
