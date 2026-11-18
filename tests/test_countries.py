"""Country table: display names, aliases, regions, prefix resolution (no network)."""
import pytest

from src.bot.countries import COUNTRIES, REGIONS, CountryMatch, CountryResolution, resolve_country
from src.profile.compiler import KNOWN_COUNTRIES


def test_table_covers_every_known_country():
    assert set(COUNTRIES) == set(KNOWN_COUNTRIES)
    for key, match in COUNTRIES.items():
        assert match.country == key
        assert match.name
        assert match.languages
        assert all(2 <= len(code) <= 3 and code.islower() for code in match.languages)
        assert "en" in match.languages


@pytest.mark.parametrize("country", sorted(KNOWN_COUNTRIES))
def test_every_known_country_resolvable(country):
    by_name = resolve_country(COUNTRIES[country].name)
    assert by_name.match == COUNTRIES[country]
    assert by_name.options == ()
    assert resolve_country(country).match == COUNTRIES[country]  # the canonical value works too


@pytest.mark.parametrize(
    "text, canonical",
    [
        ("US", "usa"),
        ("United States", "usa"),
        ("america", "usa"),
        ("UK", "uk"),
        ("Britain", "uk"),
        ("Great Britain", "uk"),
        ("England", "uk"),
        ("Czechia", "czech republic"),
        ("Holland", "netherlands"),
        ("Korea", "south korea"),
        ("Turkey", "türkiye"),
        ("UAE", "united arab emirates"),
        ("Emirates", "united arab emirates"),
        ("Deutschland", "germany"),
        ("Österreich", "austria"),
        ("Polska", "poland"),
        ("España", "spain"),
        ("Nederland", "netherlands"),
    ],
)
def test_aliases(text, canonical):
    resolution = resolve_country(text)
    assert resolution.match is not None
    assert resolution.match.country == canonical


def test_diacritics_and_case():
    for text in ("türkiye", "TURKIYE", "  Türkiye  ", "TÜRKİYE"):
        assert resolve_country(text).match.country == "türkiye"
    assert resolve_country("POLAND").match.country == "poland"


def test_ambiguous_prefix_returns_options():
    resolution = resolve_country("united")
    assert resolution.match is None
    assert {o.country for o in resolution.options} >= {"uk", "usa", "united arab emirates"}
    assert [o.name for o in resolution.options] == sorted(o.name for o in resolution.options)


def test_unique_prefix_matches():
    assert resolve_country("germ").match.country == "germany"
    assert resolve_country("switz").match.country == "switzerland"
    assert resolve_country("ind").match is None  # india and indonesia
    assert {o.country for o in resolve_country("ind").options} == {"india", "indonesia"}


def test_unknown_and_short_input():
    for text in ("xx", "atlantis", "", "  ", "po"):
        resolution = resolve_country(text)
        assert resolution == CountryResolution(None, ())


def test_regions():
    assert set(REGIONS) == {"europe", "worldwide"}
    for key, match in REGIONS.items():
        assert match.country is None
        assert match.languages == ("en",)
    assert resolve_country("Europe").match == REGIONS["europe"]
    assert resolve_country("WORLDWIDE").match.name == REGIONS["worldwide"].name
    assert resolve_country("euro").match == REGIONS["europe"]


def test_country_match_is_frozen_data():
    match = CountryMatch("poland", "Poland", ("pl", "en"))
    assert match.country == "poland"
    with pytest.raises(Exception):
        match.name = "x"  # type: ignore[misc]
