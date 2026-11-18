"""Built-in country table for ``/add_market``: English names, aliases, primary languages, regions.

``COUNTRIES`` is keyed by the canonical python-jobspy value (``KNOWN_COUNTRIES``; a test keeps the two in
step). Languages are 2-letter codes of the country's main languages plus ``en`` (used to pick the role
synonyms that apply to the market). Input is matched after folding case and diacritics.
"""
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MIN_PREFIX_CHARS = 3


@dataclass(frozen=True)
class CountryMatch:
    country: Optional[str]  # canonical python-jobspy value; None for a region
    name: str
    languages: Tuple[str, ...]


@dataclass(frozen=True)
class CountryResolution:
    match: Optional[CountryMatch]
    options: Tuple[CountryMatch, ...]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.split())


# canonical value -> (display name, primary languages); "en" is appended unless already listed.
_TABLE: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "argentina": ("Argentina", ("es",)),
    "australia": ("Australia", ("en",)),
    "austria": ("Austria", ("de",)),
    "bahrain": ("Bahrain", ("ar",)),
    "belgium": ("Belgium", ("nl", "fr", "de")),
    "brazil": ("Brazil", ("pt",)),
    "canada": ("Canada", ("en", "fr")),
    "chile": ("Chile", ("es",)),
    "china": ("China", ("zh",)),
    "colombia": ("Colombia", ("es",)),
    "costa rica": ("Costa Rica", ("es",)),
    "czech republic": ("Czech Republic", ("cs",)),
    "denmark": ("Denmark", ("da",)),
    "ecuador": ("Ecuador", ("es",)),
    "egypt": ("Egypt", ("ar",)),
    "finland": ("Finland", ("fi", "sv")),
    "france": ("France", ("fr",)),
    "germany": ("Germany", ("de",)),
    "greece": ("Greece", ("el",)),
    "hong kong": ("Hong Kong", ("zh",)),
    "hungary": ("Hungary", ("hu",)),
    "india": ("India", ("en", "hi")),
    "indonesia": ("Indonesia", ("id",)),
    "ireland": ("Ireland", ("en",)),
    "israel": ("Israel", ("he",)),
    "italy": ("Italy", ("it",)),
    "japan": ("Japan", ("ja",)),
    "kuwait": ("Kuwait", ("ar",)),
    "luxembourg": ("Luxembourg", ("fr", "de")),
    "malaysia": ("Malaysia", ("ms",)),
    "mexico": ("Mexico", ("es",)),
    "morocco": ("Morocco", ("ar", "fr")),
    "netherlands": ("Netherlands", ("nl",)),
    "new zealand": ("New Zealand", ("en",)),
    "nigeria": ("Nigeria", ("en",)),
    "norway": ("Norway", ("no",)),
    "oman": ("Oman", ("ar",)),
    "pakistan": ("Pakistan", ("ur",)),
    "panama": ("Panama", ("es",)),
    "peru": ("Peru", ("es",)),
    "philippines": ("Philippines", ("tl",)),
    "poland": ("Poland", ("pl",)),
    "portugal": ("Portugal", ("pt",)),
    "qatar": ("Qatar", ("ar",)),
    "romania": ("Romania", ("ro",)),
    "saudi arabia": ("Saudi Arabia", ("ar",)),
    "singapore": ("Singapore", ("en", "zh")),
    "south africa": ("South Africa", ("en",)),
    "south korea": ("South Korea", ("ko",)),
    "spain": ("Spain", ("es",)),
    "sweden": ("Sweden", ("sv",)),
    "switzerland": ("Switzerland", ("de", "fr", "it")),
    "taiwan": ("Taiwan", ("zh",)),
    "thailand": ("Thailand", ("th",)),
    "türkiye": ("Türkiye", ("tr",)),
    "uk": ("United Kingdom", ("en",)),
    "ukraine": ("Ukraine", ("uk",)),
    "united arab emirates": ("United Arab Emirates", ("ar",)),
    "uruguay": ("Uruguay", ("es",)),
    "usa": ("United States", ("en",)),
    "venezuela": ("Venezuela", ("es",)),
    "vietnam": ("Vietnam", ("vi",)),
}

COUNTRIES: Dict[str, CountryMatch] = {
    key: CountryMatch(key, name, languages if "en" in languages else languages + ("en",))
    for key, (name, languages) in _TABLE.items()
}

REGIONS: Dict[str, CountryMatch] = {
    "europe": CountryMatch(None, "Europe", ("en",)),
    "worldwide": CountryMatch(None, "Worldwide", ("en",)),
}

_ALIAS_SOURCE: Dict[str, str] = {
    "us": "usa", "u.s.": "usa", "united states": "usa", "united states of america": "usa", "america": "usa",
    "uk": "uk", "u.k.": "uk", "united kingdom": "uk", "britain": "uk", "great britain": "uk", "england": "uk",
    "holland": "netherlands", "nederland": "netherlands",
    "korea": "south korea", "czechia": "czech republic", "turkey": "türkiye",
    "uae": "united arab emirates", "emirates": "united arab emirates",
    # native names (EU countries)
    "deutschland": "germany", "osterreich": "austria", "österreich": "austria",
    "belgique": "belgium", "belgie": "belgium", "belgië": "belgium",
    "polska": "poland", "cesko": "czech republic", "česko": "czech republic",
    "ceska republika": "czech republic", "česká republika": "czech republic",
    "danmark": "denmark", "suomi": "finland", "espana": "spain", "españa": "spain",
    "italia": "italy", "magyarorszag": "hungary", "magyarország": "hungary",
    "norge": "norway", "sverige": "sweden", "schweiz": "switzerland", "suisse": "switzerland",
    "svizzera": "switzerland", "romania": "romania", "românia": "romania", "eire": "ireland",
    "éire": "ireland", "luxemburg": "luxembourg", "hellas": "greece", "ellada": "greece",
    "belgien": "belgium", "polen": "poland", "frankreich": "france", "niederlande": "netherlands",
}
ALIASES: Dict[str, str] = {_fold(alias): country for alias, country in _ALIAS_SOURCE.items()}


def _exact_index() -> Dict[str, CountryMatch]:
    index: Dict[str, CountryMatch] = {}
    for key, match in COUNTRIES.items():
        index[_fold(key)] = match
        index[_fold(match.name)] = match
    for alias, country in ALIASES.items():
        index[alias] = COUNTRIES[country]
    for key, region in REGIONS.items():
        index[_fold(key)] = region
        index[_fold(region.name)] = region
    return index


_EXACT = _exact_index()


def resolve_country(text: str) -> CountryResolution:
    """Exact name, alias or region; else a prefix (3+ characters) of a name or alias; else nothing."""
    folded = _fold(text)
    if not folded:
        return CountryResolution(None, ())
    exact = _EXACT.get(folded)
    if exact is not None:
        return CountryResolution(exact, ())
    if len(folded) < MIN_PREFIX_CHARS:
        return CountryResolution(None, ())
    found: List[CountryMatch] = []
    for name, match in _EXACT.items():
        if name.startswith(folded) and match not in found:
            found.append(match)
    if len(found) == 1:
        return CountryResolution(found[0], ())
    return CountryResolution(None, tuple(sorted(found, key=lambda m: m.name)))
