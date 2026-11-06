import json
import re
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "eval_cases.json"
REQUIRED = {"id", "category", "title", "company", "location", "description", "expected"}


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_loads_and_is_well_formed():
    cases = _load()
    assert 12 <= len(cases) <= 16
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert REQUIRED <= set(case), case["id"]
        assert len(case["description"]) >= 150, case["id"]
        assert len(case["description"].split()) >= 30, case["id"]
        lo, hi = case["expected"]["min"], case["expected"]["max"]
        assert 1.0 <= lo <= hi <= 10.0, case["id"]
        assert case["category"] in {"hard_reject", "relevant"}


def test_fixture_bands_match_categories():
    cases = _load()
    assert any(c["category"] == "hard_reject" for c in cases)
    assert any(c["category"] == "relevant" for c in cases)
    for case in cases:
        if case["category"] == "hard_reject":
            assert case["expected"]["max"] <= 2.0, case["id"]
        else:
            assert case["expected"]["min"] >= 6.0, case["id"]


def test_fixture_covers_required_trap_scenarios():
    cases = _load()
    trap_ids = {c["id"] for c in cases if c["category"] == "hard_reject"}
    for expected_id in (
        "trap-it-pm-coaching-side-duty",
        "trap-scrum-master",
        "trap-sales-director",
        "trap-software-engineer",
        "trap-executive-coach-native-c2",
    ):
        assert expected_id in trap_ids, expected_id


def test_fixture_has_no_home_paths_or_secrets():
    text = FIXTURE.read_text(encoding="utf-8")
    assert not re.search(r"/home/[A-Za-z0-9._-]+/", text)
    assert "file:" + "///" not in text
    assert not re.search(r"(sk-|nvapi-)[A-Za-z0-9]{10,}", text)
