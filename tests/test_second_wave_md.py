"""Tests for scripts/second_wave_md.py (markdown from results files, no recomputation)."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "second_wave_md", Path("scripts") / "second_wave_md.py"
)
assert SPEC is not None and SPEC.loader is not None
sw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sw)

RESULTS = Path("results")
BUILDERS = [
    sw.fairness_md,
    sw.unbalanced_md,
    sw.cvar_md,
    sw.noise_ablation_md,
    sw.landscape_md,
    sw.calibration_md,
]


@pytest.mark.parametrize("builder", BUILDERS)
def test_missing_inputs_degrade_to_a_note(builder, tmp_path: Path) -> None:
    out = builder(tmp_path)
    assert isinstance(out, str) and out.startswith(("No ", "Skipped"))


@pytest.mark.parametrize("builder", BUILDERS)
def test_real_inputs_render_a_table(builder) -> None:
    out = builder(RESULTS)
    assert "|---|" in out or out.startswith("Skipped")


def test_fairness_table_matches_json() -> None:
    import json

    d = json.loads((RESULTS / "price_of_fairness.json").read_text(encoding="utf-8"))
    out = sw.fairness_md(RESULTS)
    assert out.count("\n| ") >= len(d["rows"])
    worst = max(s["max_loss_pct"] for s in d["by_lambda"].values())
    assert f"{worst:.4f}" in out


def test_unbalanced_section_reports_the_failing_seeds() -> None:
    import json

    d = json.loads((RESULTS / "unbalanced_min_decode.json").read_text(encoding="utf-8"))
    out = sw.unbalanced_md(RESULTS)
    assert str(d["fail_seeds"]) in out and str(d["pass_seeds"]) in out
    assert "not tuned" in out


def test_noise_ablation_states_sign_counts() -> None:
    import json

    d = json.loads((RESULTS / "noise_ablation.json").read_text(encoding="utf-8"))
    deltas = [r["deltas"]["zne_raw_vs_noisy"] for r in d["rows"]]
    out = sw.noise_ablation_md(RESULTS)
    assert f"positive on {sum(x > 0 for x in deltas)}" in out
    assert f"negative on {sum(x < 0 for x in deltas)}" in out


def test_calibration_skip_is_reported(tmp_path: Path) -> None:
    (tmp_path / "ibm_calibration.json").write_text(
        '{"status": "skipped", "reason": "no credentials"}', encoding="utf-8"
    )
    assert "Skipped: no credentials" in sw.calibration_md(tmp_path)


def test_unbalanced_section_leads_with_the_failure_count() -> None:
    """The failure count must come before the technique line and any QAOA table."""
    import json

    d = json.loads((RESULTS / "unbalanced_min_decode.json").read_text(encoding="utf-8"))
    out = sw.unbalanced_md(RESULTS)
    lead = f"**{len(d['fail_seeds'])} of {len(d['rows'])} seeds fail"
    assert out.startswith(lead)
    assert (
        out.index(lead) < out.index("Technique:") < out.index("QAOA on both versions")
    )
