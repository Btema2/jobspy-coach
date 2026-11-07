"""Dedup hash normalization (cross-board near-duplicates) and the rehash migration."""
from pathlib import Path

from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting


def _h(company, title, location):
    return compute_dedup_hash(company, title, location)


# ---------------------------------------------------------------------------
# Shapes seen in the live run (synthetic names): these must collide.
# ---------------------------------------------------------------------------

def test_regional_branch_word_and_parenthetical_in_company_collide():
    assert _h("Acme Relief Org Polska", "Field Analyst", "Lyon, France") == _h(
        "Acme Relief Org", "Field Analyst", "Lyon, France"
    )
    assert _h("Acme Relief Org (ARO)", "Field Analyst", "Lyon, France") == _h(
        "Acme Relief Org", "Field Analyst", "Lyon, France"
    )
    assert _h("Acme Relief Org Poland Sp. z o.o.", "Field Analyst", "Lyon") == _h(
        "Acme Relief Org", "Field Analyst", "Lyon"
    )


def test_connector_word_in_title_collides():
    assert _h("Acme", "Logistics Supply Coordinator", "Lyon") == _h(
        "Acme", "Logistics and Supply Coordinator", "Lyon"
    )
    assert _h("Acme", "Health & Safety Officer", "Lyon") == _h("Acme", "Health and Safety Officer", "Lyon")
    assert _h("Acme", "Health &amp; Safety Officer", "Lyon") == _h("Acme", "Health and Safety Officer", "Lyon")
    assert _h("Acme", "Szkolenia i Rozwoj", "Lyon") == _h("Acme", "Szkolenia Rozwoj", "Lyon")


def test_location_variants_of_the_same_city_collide():
    full = _h("Acme", "Field Analyst", "Lyon, Auvergne-Rhone-Alpes, France")
    assert full == _h("Acme", "Field Analyst", "Lyon Metropolitan Area")
    assert full == _h("Acme", "Field Analyst", "Lyon, France")
    assert full == _h("Acme", "Field Analyst", "LYON")


def test_diacritics_and_gender_markers_fold():
    assert _h("Acme", "Specjalista Szkoleń (K/M)", "Besançon, Bourgogne-Franche-Comté, France") == _h(
        "Acme", "Specjalista Szkolen", "Besancon, France"
    )
    assert _h("Acme", "Łódź Coach (m/w/d)", "Łódź") == _h("Acme", "Lodz Coach", "Lodz")
    assert _h("Acme", "Coach (all genders)", "Hamburg") == _h("Acme", "Coach", "Hamburg")


# ---------------------------------------------------------------------------
# False-merge guards: all of these must stay DIFFERENT.
# ---------------------------------------------------------------------------

def test_different_roles_at_one_company_do_not_merge():
    assert _h("Acme", "Field Analyst", "Lyon") != _h("Acme", "Senior Field Analyst", "Lyon")
    assert _h("Acme", "Logistics Officer", "Lyon") != _h("Acme", "Supply Officer", "Lyon")
    # Parenthetical specialisations are meaningful and must not be dropped from titles.
    assert _h("Acme", "Advisor (Housing)", "Lyon") != _h("Acme", "Advisor (Employment)", "Lyon")
    # Language variants of the same template title are separate postings.
    assert _h("Acme", "Specialist with French and English", "Lyon") != _h(
        "Acme", "Specialist with Spanish and English", "Lyon"
    )
    # Roman-numeral levels at the end of a title are not connectors.
    assert _h("Acme", "Analyst I", "Lyon") != _h("Acme", "Analyst", "Lyon")
    assert _h("Acme", "Analyst I", "Lyon") != _h("Acme", "Analyst II", "Lyon")


def test_same_title_in_different_cities_does_not_merge():
    assert _h("Acme", "Field Analyst", "Lyon, Auvergne-Rhone-Alpes, France") != _h(
        "Acme", "Field Analyst", "Nantes, Pays de la Loire, France"
    )
    assert _h("Acme", "Field Analyst", "Lyon Metropolitan Area") != _h(
        "Acme", "Field Analyst", "Nantes Metropolitan Area"
    )
    assert _h("Acme", "Field Analyst", "Annecy, Auvergne-Rhone-Alpes, France") != _h(
        "Acme", "Field Analyst", "Grenoble, Auvergne-Rhone-Alpes, France"
    )
    # A country-level location is not collapsed into any city.
    assert _h("Acme", "Field Analyst", "France") != _h("Acme", "Field Analyst", "Lyon, France")


def test_remote_locations_keep_their_country():
    assert _h("Acme", "Coach", "Remote France") != _h("Acme", "Coach", "Remote Germany")
    assert _h("Acme", "Coach", "Remote, France") != _h("Acme", "Coach", "Remote, Germany")


def test_different_companies_do_not_merge():
    assert _h("Acme Relief Org", "Field Analyst", "Lyon") != _h("Beta Relief Org", "Field Analyst", "Lyon")
    assert _h("Acme", "Field Analyst", "Lyon") != _h("Acme Capital", "Field Analyst", "Lyon")
    # A company whose whole name is a descriptor word is not emptied.
    assert _h("Global", "Field Analyst", "Lyon") != _h("International", "Field Analyst", "Lyon")


def test_empty_fields_still_hash():
    h = _h("", "", "")
    assert isinstance(h, str) and len(h) == 64
    assert _h("(ARO)", "and", "Metropolitan Area") != _h("", "", "")


# ---------------------------------------------------------------------------
# rehash_all migration
# ---------------------------------------------------------------------------

def _job(company, title, location, old_hash, **kw):
    return JobPosting(
        dedup_hash=old_hash,
        site="linkedin",
        job_url=f"https://example.com/{old_hash}",
        title=title,
        company=company,
        location=location,
        description="d" * 300,
        **kw,
    )


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "rehash.db")
    db.init_db()
    return db


def _ids(db):
    with db._get_connection() as conn:
        return [r["id"] for r in conn.execute("SELECT id FROM jobs ORDER BY id")]


def test_rehash_all_rewrites_hashes_to_the_new_function(tmp_path):
    db = _make_db(tmp_path)
    jid, _ = db.save_job(_job("Acme Org", "Field Analyst", "Lyon", "legacy-hash-1"))

    stats = db.rehash_all()

    assert stats == {"total": 1, "rehashed": 1, "merged": 0, "kept_twins": 0}
    assert db.get_job_by_id(jid).dedup_hash == compute_dedup_hash("Acme Org", "Field Analyst", "Lyon")
    assert db.get_job_by_hash("legacy-hash-1") is None


def test_rehash_all_is_idempotent(tmp_path):
    db = _make_db(tmp_path)
    db.save_job(_job("Acme Org Polska", "Logistics Officer", "Lyon Metropolitan Area", "old-1"))
    db.save_job(_job("Acme Org", "Logistics and Officer", "Lyon, Auvergne-Rhone-Alpes, France", "old-2"))

    first = db.rehash_all()
    second = db.rehash_all()

    assert first["merged"] == 1
    assert second == {"total": 1, "rehashed": 0, "merged": 0, "kept_twins": 0}


def test_rehash_all_merges_duplicates_keeping_oldest_row(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(_job("Acme Org Polska", "Logistics Officer", "Lyon Metropolitan Area", "old-1"))
    b, _ = db.save_job(_job("Acme Org", "Logistics and Officer", "Lyon, Auvergne-Rhone-Alpes, France", "old-2"))
    other, _ = db.save_job(_job("Acme Org", "Logistics Officer", "Nantes, France", "old-3"))

    stats = db.rehash_all()

    assert stats["merged"] == 1
    assert _ids(db) == [a, other]
    assert db.get_job_by_id(b) is None


def test_rehash_all_prefers_the_row_that_already_sent_an_alert(tmp_path):
    db = _make_db(tmp_path)
    older, _ = db.save_job(_job("Acme Org Polska", "Field Analyst", "Lyon", "old-1"))
    newer, _ = db.save_job(
        _job("Acme Org", "Field Analyst", "Lyon", "old-2", eval_status="evaluated", notification_status="instant_sent")
    )

    db.rehash_all()

    assert _ids(db) == [newer]
    assert db.get_job_by_id(newer).notification_status == "instant_sent"


def test_rehash_all_prefers_an_evaluated_row_over_an_unprocessed_one(tmp_path):
    db = _make_db(tmp_path)
    older, _ = db.save_job(_job("Acme Org Polska", "Field Analyst", "Lyon", "old-1"))
    newer, _ = db.save_job(_job("Acme Org", "Field Analyst", "Lyon", "old-2", eval_status="evaluated", eval_score=7.0))

    db.rehash_all()

    assert _ids(db) == [newer]


def test_rehash_all_prefers_a_full_row_over_an_older_pre_filtered_stub(tmp_path):
    db = _make_db(tmp_path)
    stub, _ = db.save_job(
        _job("Acme Org Polska", "Field Analyst", "Lyon", "old-1", pre_filtered=True, pre_filter_reason="DESCRIPTION_TOO_SHORT")
    )
    good, _ = db.save_job(_job("Acme Org", "Field Analyst", "Lyon", "old-2"))

    stats = db.rehash_all()

    assert stats["merged"] == 1
    assert _ids(db) == [good]
    assert db.get_job_by_id(good).pre_filtered is False


def test_rehash_all_never_deletes_two_rows_that_both_sent_alerts(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(
        _job("Acme Org Polska", "Field Analyst", "Lyon", "old-1", eval_status="evaluated", notification_status="digest_sent")
    )
    b, _ = db.save_job(
        _job("Acme Org", "Field Analyst", "Lyon", "old-2", eval_status="evaluated", notification_status="instant_sent")
    )

    stats = db.rehash_all()
    again = db.rehash_all()

    assert _ids(db) == [a, b]
    assert stats["merged"] == 0 and stats["kept_twins"] == 1
    # The oldest alerted row owns the canonical hash; the twin keeps a unique filler hash.
    canonical = compute_dedup_hash("Acme Org", "Field Analyst", "Lyon")
    assert db.get_job_by_id(a).dedup_hash == canonical
    assert db.get_job_by_id(b).dedup_hash != canonical
    assert again["merged"] == 0 and again["rehashed"] == 0


def test_rehash_all_dry_run_changes_nothing(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(_job("Acme Org Polska", "Field Analyst", "Lyon", "old-1"))
    db.save_job(_job("Acme Org", "Field Analyst", "Lyon", "old-2"))

    stats = db.rehash_all(dry_run=True)

    assert stats["merged"] == 1 and stats["rehashed"] >= 1
    assert len(_ids(db)) == 2
    assert db.get_job_by_id(a).dedup_hash == "old-1"


def test_save_job_after_rehash_dedups_a_new_variant(tmp_path):
    db = _make_db(tmp_path)
    db.save_job(_job("Acme Org", "Logistics Officer", "Lyon, France", "old-1"))
    db.rehash_all()

    variant = _job(
        "Acme Org Polska", "Logistics and Officer", "Lyon Metropolitan Area",
        compute_dedup_hash("Acme Org Polska", "Logistics and Officer", "Lyon Metropolitan Area"),
    )
    _, created = db.save_job(variant)

    assert created is False
