import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from src.notifier.templates import format_digest_chunks, format_wow_alert
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        timeout: float = 15.0,
        client: Optional[httpx.Client] = None,
    ):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self._external_client = client is not None
        self._client = client or httpx.Client(timeout=self.timeout)

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Any] = None,
        timeout: float = 15.0,
        client: Optional[httpx.Client] = None,
    ) -> "TelegramNotifier":
        if settings is None:
            from src.config import get_settings

            settings = get_settings()
        return cls(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            timeout=timeout,
            client=client,
        )

    def close(self) -> None:
        if not self._external_client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "TelegramNotifier":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def send_message(self, html_text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram Bot Token or Chat ID not configured. Skipping notification.")
            return False

        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.post(self.api_url, json=payload)
                if resp.status_code == 429:
                    retry_after = 5
                    try:
                        data = resp.json()
                        if isinstance(data, dict):
                            retry_after = data.get("parameters", {}).get("retry_after", 5)
                    except Exception:
                        pass
                    if "Retry-After" in resp.headers:
                        try:
                            retry_after = int(resp.headers["Retry-After"])
                        except (ValueError, TypeError):
                            pass

                    if attempt < max_retries:
                        logger.warning(
                            f"Telegram rate limited (429). Retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(float(retry_after))
                        continue
                    else:
                        logger.error(
                            f"Telegram rate limited (429). Max retries ({max_retries}) exceeded."
                        )
                        return False

                resp.raise_for_status()
                return True
            except httpx.HTTPStatusError as e:
                logger.error(f"Telegram HTTP error: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
                return False

    def send_wow_alert(self, job: JobPosting) -> bool:
        text, markup = format_wow_alert(job)
        return self.send_message(text, reply_markup=markup)

    def send_daily_digest(
        self,
        jobs: List[JobPosting],
        date_str: str,
        update_notice: Optional[str] = None,
    ) -> int:
        if not jobs:
            return 0

        chunks = format_digest_chunks(jobs, date_str=date_str, update_notice=update_notice)
        sent_count = 0
        for chunk in chunks:
            if self.send_message(chunk):
                sent_count += 1
            # Respect 1 msg/sec pacing
            time.sleep(1.0)
        return sent_count
