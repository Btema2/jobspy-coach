import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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


# Location-neutral defaults. Personal / regional targets belong in the git-ignored queries.json.
DEFAULT_QUERY_CATALOG: List[SearchQuery] = [
    SearchQuery(
        query_id="Q-EU-REM",
        search_term="Career Coach OR Life Coach",
        location="Europe",
        # No indeed: it searches one country at a time and needs a country_code.
        site_names=["linkedin", "google"],
        is_remote=True,
    ),
    SearchQuery(
        query_id="Q-GL-REM",
        search_term="Executive Coach OR Leadership Coach",
        location="Remote",
        site_names=["linkedin", "indeed", "glassdoor", "google"],
        country_code="usa",
        is_remote=True,
    ),
]


def load_custom_queries() -> Optional[List[SearchQuery]]:
    """Loads user-specific search queries from local git-ignored queries.json if present."""
    for path in [Path("queries.json"), Path("config/queries.json")]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return [
                    SearchQuery(
                        query_id=item["query_id"],
                        search_term=item["search_term"],
                        location=item["location"],
                        site_names=item["site_names"],
                        country_code=item.get("country_code"),
                        is_remote=item.get("is_remote", False),
                        google_search_term=item.get("google_search_term"),
                    )
                    for item in data
                ]
            except Exception as e:
                logger.warning("Could not load custom queries from %s: %s", path, e)
    return None


def get_active_queries() -> List[SearchQuery]:
    """Returns custom queries if present, otherwise default catalog."""
    custom = load_custom_queries()
    return custom if custom else DEFAULT_QUERY_CATALOG
