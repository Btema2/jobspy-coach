"""Per-query job hits: upsert semantics, FK cascade and survival through rehash."""
from pathlib import Path

from src.storage.db import Database
from src.storage.models import JobPosting


def _job(company, title, location, old_hash, description="d" * 300):
    return JobPosting(
        dedup_hash=old_hash,
        site="linkedin",
        job_url=f"https://example.com/{old_hash}",
        title=title,
        company=company,
        location=location,
        description=description,
    )


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "hits.db")
    db.init_db()
    return db


def test_record_job_hit_upserts(tmp_path):
    db = _make_db(tmp_path)
    jid, _ = db.save_job(_job("Acme", "Analyst", "Lyon", "h1"))

    db.record_job_hit(jid, "q1")
    first = db.get_job_hits(jid)
    db.record_job_hit(jid, "q1")
    second = db.get_job_hits(jid)

    assert len(first) == 1 and len(second) == 1
    assert second[0]["query_key"] == "q1"
    assert second[0]["first_seen"] == first[0]["first_seen"]
    assert second[0]["last_seen"] >= second[0]["first_seen"]


def test_hits_for_two_queries_on_one_job(tmp_path):
    db = _make_db(tmp_path)
    jid, _ = db.save_job(_job("Acme", "Analyst", "Lyon", "h1"))

    db.record_job_hit(jid, "q-b")
    db.record_job_hit(jid, "q-a")

    hits = db.get_job_hits(jid)
    assert sorted(h["query_key"] for h in hits) == ["q-a", "q-b"]
    assert set(hits[0]) == {"query_key", "first_seen", "last_seen"}


def test_deleting_a_job_cascades_hits(tmp_path):
    db = _make_db(tmp_path)
    jid, _ = db.save_job(_job("Acme", "Analyst", "Lyon", "h1", description=""))
    db.record_job_hit(jid, "q1")

    assert db.delete_empty_description_jobs("linkedin") == 1

    assert db.get_job_hits(jid) == []
    with db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_hits").fetchone()[0] == 0


def test_rehash_repoints_hits_to_keeper(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(_job("Acme Org Polska", "Logistics Officer", "Lyon Metropolitan Area", "old-1"))
    b, _ = db.save_job(_job("Acme Org", "Logistics and Officer", "Lyon, Auvergne-Rhone-Alpes, France", "old-2"))
    db.record_job_hit(a, "q-a")
    db.record_job_hit(b, "q-b")

    stats = db.rehash_all()

    assert stats["merged"] == 1
    with db._get_connection() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM jobs ORDER BY id")]
    assert ids == [a]
    assert sorted(h["query_key"] for h in db.get_job_hits(a)) == ["q-a", "q-b"]
    assert db.get_job_hits(b) == []


def test_rehash_hit_key_clash_does_not_fail(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(_job("Acme Org Polska", "Logistics Officer", "Lyon Metropolitan Area", "old-1"))
    b, _ = db.save_job(_job("Acme Org", "Logistics and Officer", "Lyon, Auvergne-Rhone-Alpes, France", "old-2"))
    db.record_job_hit(a, "same")
    db.record_job_hit(b, "same")

    db.rehash_all()

    hits = db.get_job_hits(a)
    assert [h["query_key"] for h in hits] == ["same"]


def test_rehash_dry_run_keeps_hits_untouched(tmp_path):
    db = _make_db(tmp_path)
    a, _ = db.save_job(_job("Acme Org Polska", "Logistics Officer", "Lyon Metropolitan Area", "old-1"))
    b, _ = db.save_job(_job("Acme Org", "Logistics and Officer", "Lyon, Auvergne-Rhone-Alpes, France", "old-2"))
    db.record_job_hit(a, "q-a")
    db.record_job_hit(b, "q-b")

    db.rehash_all(dry_run=True)

    assert [h["query_key"] for h in db.get_job_hits(a)] == ["q-a"]
    assert [h["query_key"] for h in db.get_job_hits(b)] == ["q-b"]
