import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bakeoff.py"
spec = importlib.util.spec_from_file_location("bakeoff", SCRIPT)
bakeoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bakeoff)

VALID = (
    '{"score": 1.5, "summary_uk": "Коротке резюме вакансії.", "reasoning": "r", '
    '"language_requirements": [], "pros": [], "cons": [], "hard_reject": true}'
)


def rec(case_id, category, score, valid=True, in_band=True, **kw):
    base = {
        "case_id": case_id, "category": category, "status": 200, "valid_json": valid,
        "score": score, "in_band": in_band, "latency_s": 10.0, "finish_reason": "stop",
    }
    base.update(kw)
    return base


def test_strip_think_removes_blocks_and_unopened_tail():
    assert bakeoff.strip_think("<think>abc</think>{\"a\": 1}") == '{"a": 1}'
    assert bakeoff.strip_think("reasoning text</think>\n{\"a\": 1}") == '{"a": 1}'
    assert bakeoff.strip_think("") == ""


def test_score_in_band_edges_and_none():
    band = {"min": 6.0, "max": 10.0}
    assert bakeoff.score_in_band(6.0, band)
    assert not bakeoff.score_in_band(5.9, band)
    assert not bakeoff.score_in_band(None, band)


def test_percentile_nearest_rank():
    assert bakeoff.percentile([], 95) is None
    assert bakeoff.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert bakeoff.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5


def test_parse_model_output_handles_fence_think_and_garbage():
    ok = bakeoff.parse_model_output("<think>x</think>```json\n" + VALID + "\n```")
    assert ok["valid_json"] and ok["score"] == 1.5 and ok["had_fence"] and ok["had_think"]
    assert ok["hard_reject"] is True
    bad = bakeoff.parse_model_output("User Safety: safe")
    assert not bad["valid_json"] and bad["score"] is None
    assert not bakeoff.parse_model_output(None)["valid_json"]


def test_summarize_and_decision_rules():
    good = [
        rec("t1", "hard_reject", 1.0), rec("t2", "hard_reject", 2.0),
        rec("p1", "relevant", 8.0), rec("p2", "relevant", 7.0),
        rec("p3", "relevant", 7.5), rec("p4", "relevant", 4.0, in_band=False),
    ]
    s = bakeoff.summarize(good)
    assert s["trap_max_score"] == 2.0 and s["positive_share"] == 0.75
    assert not bakeoff.passes_nim_rule(s)  # 75% < 80%
    good[-1] = rec("p4", "relevant", 8.0)
    assert bakeoff.passes_nim_rule(bakeoff.summarize(good))


def test_trap_above_limit_or_invalid_json_fails():
    recs = [rec("t1", "hard_reject", 2.5), rec("p1", "relevant", 8.0)]
    s = bakeoff.summarize(recs)
    assert s["trap_failures"] == ["t1"] and not bakeoff.passes_nim_rule(s)
    assert bakeoff.backup_qualifies(bakeoff.summarize(recs, trap_limit=3.0))
    invalid = [rec("t1", "hard_reject", None, valid=False), rec("p1", "relevant", 8.0)]
    assert not bakeoff.backup_qualifies(bakeoff.summarize(invalid, trap_limit=3.0))


def test_backup_rejects_trap_over_three():
    s = bakeoff.summarize([rec("t1", "hard_reject", 3.5), rec("p1", "relevant", 8.0)], trap_limit=3.0)
    assert not bakeoff.backup_qualifies(s)


def test_workers_for_sizing_rule():
    assert bakeoff.workers_for(30, 15, margin=0) == 8
    assert bakeoff.workers_for(30, 90, margin=0) == 45
    assert bakeoff.workers_for(30, 0.1) >= 1


def test_free_model_exclusions():
    base = {"context_length": 128000, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}
    assert bakeoff.is_excluded_free_model({**base, "id": "x/llama-guard-3:free", "name": "Guard"}) == "content-safety"
    assert bakeoff.is_excluded_free_model({**base, "id": "x/some-coder:free", "name": "Some Coder"}) == "code"
    assert bakeoff.is_excluded_free_model({**base, "id": "x/m:free", "name": "M", "expiration_date": "2026-12-01"})
    assert bakeoff.is_excluded_free_model({**base, "id": "x/tiny-3b:free", "name": "Tiny 3B"}) == "tiny"
    assert bakeoff.is_excluded_free_model({**base, "id": "openrouter/free", "name": "Free Router"}) == "router"
    assert bakeoff.is_excluded_free_model({**base, "id": "x/gemma-31b-it:free", "name": "Gemma 31B"}) is None


def test_free_model_non_text_modalities_excluded():
    big = {"id": "x/big-model:free", "name": "Big Model", "context_length": 128000}
    image_out = {"input_modalities": ["text"], "output_modalities": ["text", "image"]}
    audio_only = {"input_modalities": ["audio"], "output_modalities": ["text"]}
    text_only = {"input_modalities": ["text", "image"], "output_modalities": ["text"]}
    assert bakeoff.is_excluded_free_model({**big, "architecture": image_out}) == "non-text output"
    assert bakeoff.is_excluded_free_model({**big, "architecture": audio_only}) == "non-text output"
    assert bakeoff.is_excluded_free_model({**big, "architecture": text_only}) is None


def test_load_env_value_reads_file_without_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_KEY='abc'\nOTHER=1\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    assert bakeoff.load_env_value("FOO_KEY", env_file=env) == "abc"
    assert bakeoff.load_env_value("MISSING_KEY", env_file=env) is None
