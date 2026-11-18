from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    search_term: str
    location: str
    site_names: List[str]
    country_code: Optional[str] = None
    is_remote: bool = False
    # Optional full Google Jobs query override. None -> derived per query by the scraper.
    google_search_term: Optional[str] = None


def query_key(query: SearchQuery, board: str) -> str:
    """Stable identifier of one (query, board) pair, used to key per-query scrape history."""
    return f"{query.query_id}|{board.lower()}"
