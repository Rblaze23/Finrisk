"""Report writer: facts, rule-based verdicts, guarded LLM sections, rendering."""
import json

import pytest

from finrisk.genai import report_writer as RW
from finrisk.genai.number_guard import check_text
from finrisk.validation.results import make, write_json

C, X = "scorecard_independent", "xgboost_independent"


def _deciles():
    return [{"bin": i, "n": 1000, "n_defaults": 100, "mean_pd": 0.05 + 0.02 * i, "observed_dr": 0.06 + 0.025 * i,
             "ratio": 1.2, "ci_low": 0.04 + 0.025 * i, "ci_high": 0.08 + 0.025 * i, "pd_inside_ci": False}
            for i in range(10)]


def _model_rows(m, role, gini=(0.295, 0.293, 0.298), ratio=1.204, cal_status="fail", extra=()):
    rows = [
        make(m, "calibration", "valid_calibration_ratio", 1.07, "warn", 0.1, 0.2),
        make(m, "calibration", "oot_calibration_ratio", ratio, cal_status, 0.1, 0.2, "rule",
             {"mean_pd": 0.1236, "observed_dr": 0.1489, "ratio": ratio, "deciles": _deciles()}),
        make(m, "calibration", "oot_calibration_slope", 0.95, "pass", 0.1, 0.25),
        make(m, "calibration", "valid_calibration_slope", 0.96, "pass", 0.1, 0.25),
        make(m, "calibration", "oot_brier", 0.123, "info"),
        make(m, "calibration", "oot_bands_underpredicted", 5, "fail", 1, 3, "rule", {"bands": [{"band": f"R{i}"} for i in range(1, 7)]}),
        make(m, "out_of_time", "train_gini", gini[0], "info", details={"gini": gini[0], "ks": 0.21}),
        make(m, "out_of_time", "valid_gini", gini[1], "info", details={"gini": gini[1], "ks": 0.21}),
        make(m, "out_of_time", "oot_gini", gini[2], "info", details={"gini": gini[2], "ks": 0.214}),
        make(m, "out_of_time", "gini_drop_train_to_oot", (gini[0] - gini[2]) / gini[0], "pass", 0.1, 0.2),
        make(m, "out_of_time", "gini_drop_valid_to_oot", (gini[1] - gini[2]) / gini[1], "pass", 0.1, 0.2),
        make(m, "out_of_time", "worst_quarter_gini", 0.28, "pass", 0.25, 0.15),
        make(m, "out_of_time", "oot_gini_floor", gini[2], "pass", 0.25, 0.15),
        make(m, "recalibration", "intercept_oot_calibration_ratio", 1.126, "warn", 0.1, 0.2),
        make(m, "recalibration", "platt_oot_calibration_ratio", 1.125, "warn", 0.1, 0.2),
        make(m, "recalibration", "isotonic_oot_calibration_ratio", 1.125, "warn", 0.1, 0.2),
        make(m, "stability", "score_psi_train_to_valid", 0.003, "pass", 0.1, 0.25),
        make(m, "stability", "score_psi_train_to_oot", 0.012, "pass", 0.1, 0.25),
        make(m, "stability", "score_psi_train_to_monitor", 0.054, "pass", 0.1, 0.25),
        make(m, "stability", "csi_max_used_feature_train_to_oot", 1.197, "fail", 0.1, 0.25, "rule",
             {"worst_feature": "total_rev_hi_lim", "n_used_above_warn": 3, "n_used_features": 11}),
        make(m, "stability", "default_rate_shift_train_to_oot", 0.172, "info", details={"train_dr": 0.127, "oot_dr": 0.1489}),
        make(m, "sensitivity", "vintage_sensitive_features", 6, "info", details={"features": ["avg_cur_bal", "tot_cur_bal"]}),
        make(m, "sensitivity", "oot_gini_without_vintage_features", 0.29, "info", details={"gini_original": gini[2], "gini_reduced": 0.29}),
        make(m, "sensitivity", "vintage_gini_loss", 0.027, "pass", 0.03, 0.1),
        make(m, "leakage", "post_origination_features_in_model", 0, "pass", 0, 0),
        make(m, "leakage", "max_single_feature_gini", 0.25, "pass", 0.4, 0.5, "rule", {"strongest": "fico_mid"}),
        make(m, "leakage", "demonstration_leaky_model_oot_gini", 0.95, "info", details={"clean_model_gini": gini[2]}),
        make(m, "explanation_stability", "importance_rank_corr_train_to_oot", 0.97, "pass", 0.8, 0.6, "rule", {"top5_overlap": 0.8}),
        make(m, "explanation_stability", "importance_rank_corr_train_to_valid", 0.98, "pass", 0.8, 0.6, "rule", {"top5_overlap": 1.0}),
        make(m, "fairness", "oot_group_calibration_dev_emp_length", 0.221, "fail", 0.1, 0.2, "rule", {"worst_group": "unknown"}),
        make(m, "fairness", "oot_group_calibration_dev_addr_state", 0.14, "warn", 0.1, 0.2, "rule", {"worst_group": "NY"}),
        make(m, "fairness", "geographic_importance_share", 0.0 if role == "champion" else 0.046,
             "pass" if role == "champion" else "warn", 0.02, 0.05),
        make(m, "stability_detail", "csi_total_rev_hi_lim_train_to_oot", 1.2, "fail", 0.1, 0.25),
    ]
    return rows + list(extra)


def _validation(tmp_path, champ_extra=(), chall_gini=(0.403, 0.334, 0.335), chall_extra=()):
    rows = _model_rows(C, "champion", extra=champ_extra)
    rows += _model_rows(X, "challenger", gini=chall_gini, ratio=1.187, cal_status="warn", extra=chall_extra)
    rows.append(make(X, "explanation_stability", "importance_rank_corr_champion_vs_challenger_oot", 0.542, "fail", 0.8, 0.6, "rule",
                     {"top5_overlap": 0.6, "champion_top5": ["fico_mid"], "challenger_top5": ["fico_mid", "addr_state"]}))
    write_json(tmp_path / "validation_results.json", "val_test", rows,
               {C: {"role": "champion", "version": "1", "kind": "scorecard", "feature_set": "independent"},
                X: {"role": "challenger", "version": "2", "kind": "xgboost", "feature_set": "independent"}})
    return json.loads((tmp_path / "validation_results.json").read_text())


def _monitoring(tmp_path):
    months = [f"2014-{m:02d}-01 00:00:00" for m in range(1, 13)] + [f"2015-{m:02d}-01 00:00:00" for m in range(1, 13)]
    monthly = [{"month": mo, "n_loans": 11000 + 800 * i, "mean_pd": 0.128 - 0.0003 * i,
                "observed_default_rate": 0.125 + 0.0015 * i, "outcomes_available": True,
                "calibration_ratio_roll3": 1.0 + 0.01 * i, "calibration_ratio_recal_roll3": 1.0 + 0.008 * i}
               for i, mo in enumerate(months)]
    det = {"score_psi": {"first_amber_or_red": None, "first_red": None, "months_evaluated": 36, "months_red": 0, "months_amber": 0},
           "csi": {"first_amber_or_red": {"vintage": "2014-01", "visible_from": "2014-01"}, "first_red": {"vintage": "2014-01", "visible_from": "2014-01"},
                   "months_evaluated": 36, "months_red": 36, "months_amber": 0},
           "calibration": {"first_amber_or_red": {"vintage": "2014-10", "visible_from": "2015-10"},
                           "first_red": {"vintage": "2015-05", "visible_from": "2016-05"}, "months_evaluated": 22, "months_red": 5, "months_amber": 10},
           "gini": {"first_amber_or_red": None, "first_red": None, "months_evaluated": 22, "months_red": 0, "months_amber": 0}}
    rec = {"months_compared": 8, "mean_abs_deviation_raw": 0.207, "mean_abs_deviation_recalibrated": 0.152,
           "last_ratio_raw": 1.226, "last_ratio_recalibrated": 1.119, "last_month": "2015-12"}
    mon = {"params": {"outcome_lag_months": 12, "reference": "train", "outcomes_known_until": "2015-12", "recal_window": 6},
           "models": {n: {"role": r, "monthly": monthly, "detection": det, "recalibration": rec}
                      for n, r in [(C, "champion"), (X, "challenger")]}}
    (tmp_path / "monitoring_results.json").write_text(json.dumps(mon))
    return mon


@pytest.fixture()
def inputs(tmp_path):
    return _validation(tmp_path), _monitoring(tmp_path), tmp_path


class FakeLLM:
    model = "fake-model"

    def __init__(self, drafts):
        self.drafts, self.calls, self.seen = list(drafts), 0, []

    def chat(self, messages, temperature=0.0, max_tokens=700):
        self.calls += 1
        self.seen.append(messages)
        return self.drafts.pop(0) if self.drafts else self.drafts_default

    drafts_default = "The champion has a Gini of 0.298."


# ------------------------------------------------------------------------------ verdicts
def test_verdict_approved_with_conditions_when_only_remediable_fails(inputs):
    val, mon, _ = inputs
    v = RW.decide(RW.Results(val))
    assert v["champion"]["verdict"] == "Approved with conditions"
    assert v["champion"]["blocking_findings"] == []


def test_leakage_failure_is_a_blocker(tmp_path):
    bad = make(C, "leakage", "post_origination_features_in_model", 2, "fail", 0, 0)
    val = _validation(tmp_path, champ_extra=[bad])
    # the later duplicate row wins in the lookup, so the blocker is seen
    v = RW.decide(RW.Results(val))
    assert v["champion"]["verdict"] == "Not approved"
    assert "post_origination_features_in_model" in v["champion"]["blocking_findings"]


def test_low_gini_floor_failure_is_a_blocker(tmp_path):
    bad = make(C, "out_of_time", "oot_gini_floor", 0.10, "fail", 0.25, 0.15)
    v = RW.decide(RW.Results(_validation(tmp_path, champ_extra=[bad])))
    assert v["champion"]["verdict"] == "Not approved"


def test_challenger_is_not_recommended_when_gain_is_small_or_new_findings_exist(inputs):
    val, _, _ = inputs
    rep = RW.decide(RW.Results(val))["replacement"]
    assert rep["challenger_recommended_to_replace_champion"] is False
    assert any("materiality" in r for r in rep["reasons"])


def test_challenger_with_big_gain_and_no_new_findings_may_replace(tmp_path):
    val = _validation(tmp_path, chall_gini=(0.45, 0.40, 0.40))
    # remove the challenger-only flagged findings so that nothing new is raised
    val["results"] = [r for r in val["results"] if not (r["model"] == X and r["test"] in ("fairness", "explanation_stability")
                                                         and r["status"] in ("warn", "fail"))]
    for r in val["results"]:
        if r["model"] == X and r["test"] == "out_of_time" and r["status"] in ("warn", "fail"):
            r["status"] = "pass"
    rep = RW.decide(RW.Results(val))["replacement"]
    assert rep["challenger_recommended_to_replace_champion"] is True and rep["reasons"] == []


def test_conditions_are_derived_from_flagged_metrics(inputs):
    val, _, _ = inputs
    conds = RW.conditions(RW.Results(val))
    text = " ".join(c["condition"] for c in conds)
    assert "Recalibrate" in text and "leading-indicator" in text
    assert all(c["triggered_by"] for c in conds)


# --------------------------------------------------------------------------------- facts
def test_facts_contain_rounded_values_and_derived_numbers(inputs):
    val, mon, _ = inputs
    f = RW.build_facts(val, mon)
    assert f["discrimination"]["champion"]["gini_oot"] == 0.298
    assert f["discrimination"]["challenger_gini_gain_oot"] == pytest.approx(0.037, abs=1e-9)
    assert f["calibration"]["champion"]["oot_observed_default_rate_pct"] == 14.9
    assert f["calibration"]["champion"]["risk_bands_underpredicted"] == 5
    assert f["monitoring"]["champion"]["detection"]["score_psi"]["first_alert_vintage"] == "never"
    pf = f["monitoring"]["champion"]["portfolio"]
    # loans of vintages after 2014-10 up to and including 2015-10
    expected = sum(11000 + 800 * i for i in range(10, 22))
    assert pf["loans_issued_between_first_calibration_alert_and_its_visibility"] == expected


def test_missing_metric_gives_a_clear_error(inputs):
    val, mon, _ = inputs
    val["results"] = [r for r in val["results"] if r["metric"] != "oot_gini"]
    with pytest.raises(RW.MissingResult, match="run_validation"):
        RW.build_facts(val, mon)


# ----------------------------------------------------------------------- templates / guard
def test_every_template_passes_the_number_guard_against_its_own_facts(inputs):
    """Invariant: the fallback text can never introduce an unsupported number."""
    val, mon, _ = inputs
    facts = RW.build_facts(val, mon)
    for key in RW.LLM_SECTIONS:
        subset = facts[key] if "meta" in facts[key] else {"meta": facts["meta"], **facts[key]}
        text = RW.TEMPLATES[key](facts)
        res = check_text(text, subset)
        assert res.ok, (key, res.violations)


def test_llm_section_accepted_on_first_try(inputs):
    val, mon, _ = inputs
    facts = RW.build_facts(val, mon)
    llm = FakeLLM(["The champion reaches a Gini of 0.298 out-of-time, against 0.295 in training."])
    s = RW.write_section("discrimination", facts, llm)
    assert s.source == "llm" and s.attempts == 1 and s.rejected == []
    assert llm.seen[0][0]["role"] == "system" and "FACTS (JSON)" in llm.seen[0][1]["content"]


def test_invented_number_is_rejected_then_corrected(inputs):
    val, mon, _ = inputs
    facts = RW.build_facts(val, mon)
    llm = FakeLLM(["The champion reaches a Gini of 0.41.", "The champion reaches a Gini of 0.298."])
    s = RW.write_section("discrimination", facts, llm)
    assert s.source == "llm" and s.attempts == 2
    assert s.rejected[0]["violations"][0]["number"] == "0.41"
    assert "0.41" in llm.seen[1][-1]["content"]                 # feedback names the offending number


def test_section_falls_back_to_template_after_max_attempts(inputs):
    val, mon, _ = inputs
    facts = RW.build_facts(val, mon)
    llm = FakeLLM(["Gini 0.99.", "Gini 0.98.", "Gini 0.97."])
    s = RW.write_section("discrimination", facts, llm, max_attempts=3)
    assert s.source == "template" and s.attempts == 3 and len(s.rejected) == 3
    assert "0.99" not in s.text and "0.298" in s.text


def test_the_llm_cannot_see_numbers_from_other_sections(inputs):
    val, mon, _ = inputs
    facts = RW.build_facts(val, mon)
    # 5 risk bands under-predicted belongs to the calibration facts, not to discrimination
    llm = FakeLLM(["The champion has a Gini of 0.298 and 5 bands were affected."] * 3)
    s = RW.write_section("discrimination", facts, llm, max_attempts=2)
    assert s.source == "template"


# ----------------------------------------------------------------------------- end to end
def test_full_report_without_llm(inputs):
    val, mon, d = inputs
    res = RW.generate(d, llm=None)
    md = res["markdown"]
    for heading in ["Executive summary", "Calibration", "Findings and conditions", "Appendix B"]:
        assert heading in md
    assert "Approved with conditions" in md and "**not** recommended" in md
    assert "None" not in md.replace("None.", "") and "nan" not in md.lower().replace("finance", "")
    assert (d / "validation_report.md").exists() and (d / "validation_report_audit.json").exists()
    assert (d / "figures" / "calibration_deciles.png").exists() and (d / "figures" / "monitoring.png").exists()
    audit = json.loads((d / "validation_report_audit.json").read_text())
    assert all(s["source"] == "template" for s in audit["sections"])


def test_full_report_with_llm_logs_rejections_in_the_audit(inputs):
    val, mon, d = inputs
    good = {"executive_summary": "The champion reaches a Gini of 0.298 and the challenger 0.335."}
    llm = FakeLLM(["The champion reaches 0.77.", good["executive_summary"]])   # first draft invented a number
    res = RW.generate(d, llm=llm, max_attempts=2, with_figures=False)
    audit = res["audit"]
    es = next(s for s in audit["sections"] if s["section"] == "executive_summary")
    assert es["source"] == "llm" and es["attempts"] == 2 and es["rejected"][0]["violations"][0]["number"] == "0.77"
    assert audit["llm"]["model"] == "fake-model"
    assert "fake-model" in res["markdown"]


def test_html_is_self_contained_with_embedded_images(inputs):
    pytest.importorskip("markdown")
    val, mon, d = inputs
    html = RW.generate(d, llm=None)["html"]
    assert html.startswith("<!doctype html>") and "<table>" in html
    assert "data:image/png;base64," in html and 'src="figures/' not in html
