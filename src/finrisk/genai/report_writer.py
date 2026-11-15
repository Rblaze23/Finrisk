"""Evidence-based validation report.

    python -m finrisk.genai.report_writer --no-llm      # deterministic text, no API key needed
    python -m finrisk.genai.report_writer               # LLM prose (Groq), guarded
    python -m finrisk.genai.report_writer --model <id> --max-attempts 3

Inputs : reports/validation_results.json, reports/monitoring_results.json
Outputs: reports/validation_report.md, reports/validation_report.html (open it and print to PDF),
         reports/figures/*.png, reports/validation_report_audit.json

Design rules (the point of this module):
  1. Verdicts, conditions and findings tables are RULE-BASED. The LLM never decides anything.
  2. The LLM only writes prose for six sections, from a facts dictionary, at temperature 0.
  3. Every number in its prose is checked by the number guard against those facts. On a violation
     the draft is sent back with the offending numbers; after max attempts the section falls back
     to a deterministic template. The final report therefore contains only verified numbers.
  4. The audit file records, per section, who wrote it, how many attempts, and what was rejected.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from finrisk.config import get_settings
from finrisk.genai.number_guard import check_text

KIND_NAMES = {"scorecard": "WoE logistic-regression scorecard",
              "xgboost": "XGBoost gradient-boosting model",
              "lightgbm": "LightGBM gradient-boosting model"}
MATERIAL_GINI_GAIN = 0.05      # a challenger must beat the champion by this much to replace it
LLM_SECTIONS = ["executive_summary", "discrimination", "calibration", "stability",
                "explainability_fairness", "monitoring"]
TITLES = {"executive_summary": "Executive summary", "discrimination": "Discriminatory power",
          "calibration": "Calibration", "stability": "Stability, sensitivity and leakage",
          "explainability_fairness": "Explainability and fairness", "monitoring": "Monitoring"}


class MissingResult(KeyError):
    pass


# =============================================================================== inputs
class Results:
    """Lookup helpers over validation_results.json."""

    def __init__(self, val: dict):
        self.val = val
        self.rows = {(r["model"], r["metric"]): r for r in val["results"]}
        self.roles = {info["role"]: name for name, info in val["models"].items()}

    def row(self, model: str, metric: str, required: bool = True):
        r = self.rows.get((model, metric))
        if r is None and required:
            raise MissingResult(f"'{metric}' missing for {model}. Run: python -m finrisk.validation.run_validation")
        return r

    def v(self, model, metric, required=True):
        r = self.row(model, metric, required)
        return None if r is None else r["value"]

    def d(self, model, metric, required=True) -> dict:
        r = self.row(model, metric, required)
        return {} if r is None else (r["details"] or {})

    def status(self, model, metric):
        r = self.row(model, metric, False)
        return None if r is None else r["status"]

    def headline(self, model):
        return [r for r in self.val["results"] if r["model"] == model and not r["test"].endswith("_detail")]


def load_inputs(reports_dir: Path) -> tuple[dict, dict]:
    paths = [reports_dir / "validation_results.json", reports_dir / "monitoring_results.json"]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"{p} not found. Run the validation suite and the monitoring simulation first.")
    return tuple(json.loads(p.read_text(encoding="utf-8")) for p in paths)  # type: ignore[return-value]


def _r(x, nd):
    return None if x is None else round(float(x), nd)


def _pct(x):
    return None if x is None else round(100.0 * float(x), 1)


# ================================================================================ facts
def _model_facts(R: Results, name: str, kind: str) -> dict:
    g = lambda split: R.d(name, f"{split}_gini")
    cal = R.d(name, "oot_calibration_ratio")
    bands = R.d(name, "oot_bands_underpredicted").get("bands", [])
    stab = R.d(name, "csi_max_used_feature_train_to_oot")
    shift = R.d(name, "default_rate_shift_train_to_oot")
    expl = R.d(name, "importance_rank_corr_train_to_oot")
    vint = R.d(name, "vintage_sensitive_features")
    vint_g = R.d(name, "oot_gini_without_vintage_features", required=False)
    leak = R.d(name, "demonstration_leaky_model_oot_gini")
    strongest = R.d(name, "max_single_feature_gini")

    f = {"description": KIND_NAMES.get(kind, kind)}
    f["discrimination"] = {
        "gini_train": _r(g("train")["gini"], 3), "gini_valid": _r(g("valid")["gini"], 3),
        "gini_oot": _r(g("oot")["gini"], 3), "ks_oot": _r(g("oot")["ks"], 3),
        "gini_change_train_to_oot_pct": _pct(-R.v(name, "gini_drop_train_to_oot")),
        "gini_change_valid_to_oot_pct": _pct(-R.v(name, "gini_drop_valid_to_oot")),
        "worst_quarter_gini": _r(R.v(name, "worst_quarter_gini"), 3),
    }
    f["calibration"] = {
        "oot_observed_default_rate_pct": _pct(cal["observed_dr"]),
        "oot_mean_predicted_pd_pct": _pct(cal["mean_pd"]),
        "oot_calibration_ratio": _r(R.v(name, "oot_calibration_ratio"), 3),
        "oot_calibration_ratio_status": R.status(name, "oot_calibration_ratio"),
        "valid_calibration_ratio": _r(R.v(name, "valid_calibration_ratio"), 3),
        "oot_calibration_slope": _r(R.v(name, "oot_calibration_slope"), 3),
        "oot_brier_score": _r(R.v(name, "oot_brier"), 4),
        "risk_bands_underpredicted": int(R.v(name, "oot_bands_underpredicted")),
        "risk_bands_evaluated": len(bands),
        "ratio_after_recalibrating_on_validation_year": {
            m: _r(R.v(name, f"{m}_oot_calibration_ratio"), 3) for m in ("intercept", "platt", "isotonic")},
    }
    f["stability"] = {
        "score_psi_valid": _r(R.v(name, "score_psi_train_to_valid"), 3),
        "score_psi_oot": _r(R.v(name, "score_psi_train_to_oot"), 3),
        "score_psi_monitor": _r(R.v(name, "score_psi_train_to_monitor"), 3),
        "worst_input_csi_oot": _r(R.v(name, "csi_max_used_feature_train_to_oot"), 3),
        "worst_input_csi_feature": stab.get("worst_feature"),
        "model_inputs_with_csi_above_amber": int(stab.get("n_used_above_warn", 0)),
        "model_inputs_total": int(stab.get("n_used_features", 0)),
        "train_default_rate_pct": _pct(shift.get("train_dr")),
        "oot_default_rate_pct": _pct(shift.get("oot_dr")),
        "inputs_with_changed_availability": list(vint.get("features", [])),
        "n_inputs_with_changed_availability": int(R.v(name, "vintage_sensitive_features")),
        "gini_oot_without_those_inputs": _r(vint_g.get("gini_reduced", vint_g.get("gini_original")), 3)
        if vint_g else None,
        "relative_gini_loss_without_those_inputs_pct": _pct(R.v(name, "vintage_gini_loss", required=False)),
        "post_origination_inputs_in_model": int(R.v(name, "post_origination_features_in_model")),
        "strongest_single_input": strongest.get("strongest"),
        "strongest_single_input_gini": _r(R.v(name, "max_single_feature_gini"), 3),
        "deliberately_leaky_model_gini_oot": _r(R.v(name, "demonstration_leaky_model_oot_gini"), 3),
        "clean_model_gini_oot": _r(leak.get("clean_model_gini"), 3),
    }
    groups = {}
    for (m, metric), row in R.rows.items():
        if m == name and metric.startswith("oot_group_calibration_dev_"):
            groups[metric.replace("oot_group_calibration_dev_", "")] = {
                "deviation_pct": _pct(row["value"]), "worst_group": (row["details"] or {}).get("worst_group"),
                "status": row["status"]}
    cc = R.d(name, "importance_rank_corr_champion_vs_challenger_oot", required=False)
    f["explainability_fairness"] = {
        "importance_rank_correlation_train_to_oot": _r(R.v(name, "importance_rank_corr_train_to_oot"), 2),
        "top5_driver_overlap_train_to_oot": int(round(5 * expl.get("top5_overlap", 0))),
        "importance_rank_correlation_vs_champion": _r(R.v(name, "importance_rank_corr_champion_vs_challenger_oot", False), 2),
        "top5_driver_overlap_vs_champion": int(round(5 * cc["top5_overlap"])) if cc else None,
        "top5_drivers": cc.get("challenger_top5") if cc else None,
        "champion_top5_drivers": cc.get("champion_top5") if cc else None,
        "group_calibration": groups,
        "geographic_importance_share_pct": _pct(R.v(name, "geographic_importance_share")),
    }
    return f


def _monitoring_facts(mon: dict, name: str) -> dict:
    m = mon["models"][name]
    monthly = m["monthly"]
    with_out = [r for r in monthly if r.get("outcomes_available")]
    first, last_o = monthly[0], with_out[-1]
    det = {}
    for monitor, d in m["detection"].items():
        fa, fr = d["first_amber_or_red"], d["first_red"]
        det[monitor] = {"first_alert_vintage": fa["vintage"] if fa else "never",
                        "alert_visible_from": fa["visible_from"] if fa else "never",
                        "first_red_vintage": fr["vintage"] if fr else "never",
                        "red_months": d["months_red"], "amber_months": d["months_amber"],
                        "months_evaluated": d["months_evaluated"]}
    cal = m["detection"]["calibration"]["first_amber_or_red"]
    between = None
    if cal:
        between = int(sum(r["n_loans"] for r in monthly
                          if cal["vintage"] < str(r["month"])[:7] <= cal["visible_from"]))
    rec = m.get("recalibration") or {}
    return {
        "detection": det,
        "recalibration_lagged": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()},
        "portfolio": {
            "first_month": str(first["month"])[:7], "last_month_with_outcomes": str(last_o["month"])[:7],
            "mean_predicted_pd_first_month_pct": _pct(first["mean_pd"]),
            "mean_predicted_pd_last_outcome_month_pct": _pct(last_o["mean_pd"]),
            "observed_default_rate_first_month_pct": _pct(first["observed_default_rate"]),
            "observed_default_rate_last_outcome_month_pct": _pct(last_o["observed_default_rate"]),
            "monthly_volume_first_month": int(first["n_loans"]),
            "monthly_volume_last_outcome_month": int(last_o["n_loans"]),
            "volume_growth_multiple": round(last_o["n_loans"] / first["n_loans"], 1),
            "loans_issued_between_first_calibration_alert_and_its_visibility": between,
        },
    }


def build_facts(val: dict, mon: dict) -> dict:
    R = Results(val)
    champ, chall = R.roles["champion"], R.roles["challenger"]
    kinds = {n: i["kind"] for n, i in val["models"].items()}
    cf, xf = _model_facts(R, champ, kinds[champ]), _model_facts(R, chall, kinds[chall])
    verdicts = decide(R)
    p = mon["params"]

    facts = {
        "meta": {"run_id": val["run_id"], "validation_generated_at": val["generated_at"],
                 "champion": {"model": champ, "version": val["models"][champ]["version"], "description": cf["description"]},
                 "challenger": {"model": chall, "version": val["models"][chall]["version"], "description": xf["description"]},
                 "loan_term_months": 36, "train_years": "2012-2013", "validation_year": 2014,
                 "out_of_time_year": 2015, "monitoring_year": 2016,
                 "monitoring_outcome_lag_months": p["outcome_lag_months"],
                 "monitoring_reference_population": p["reference"]},
        "verdicts": verdicts,
    }
    facts["discrimination"] = {
        "champion": cf["discrimination"], "challenger": xf["discrimination"],
        "challenger_gini_gain_oot": _r(xf["discrimination"]["gini_oot"] - cf["discrimination"]["gini_oot"], 3),
        "materiality_threshold_gini_gain": MATERIAL_GINI_GAIN}
    facts["calibration"] = {"champion": cf["calibration"], "challenger": xf["calibration"],
                            "tolerance_amber_pct": 10, "tolerance_red_pct": 20}
    facts["stability"] = {"champion": cf["stability"], "challenger": xf["stability"]}
    facts["explainability_fairness"] = {"champion": cf["explainability_fairness"],
                                        "challenger": xf["explainability_fairness"]}
    facts["monitoring"] = {"champion": _monitoring_facts(mon, champ), "challenger": _monitoring_facts(mon, chall)}
    facts["executive_summary"] = {
        "verdicts": verdicts,
        "key_numbers": {
            "champion_gini_oot": cf["discrimination"]["gini_oot"], "challenger_gini_oot": xf["discrimination"]["gini_oot"],
            "champion_calibration_ratio_oot": cf["calibration"]["oot_calibration_ratio"],
            "challenger_calibration_ratio_oot": xf["calibration"]["oot_calibration_ratio"],
            "champion_score_psi_oot": cf["stability"]["score_psi_oot"],
            "champion_calibration_first_alert": facts["monitoring"]["champion"]["detection"]["calibration"],
            "champion_score_psi_alert": facts["monitoring"]["champion"]["detection"]["score_psi"],
            "loans_issued_before_calibration_alert_visible":
                facts["monitoring"]["champion"]["portfolio"]["loans_issued_between_first_calibration_alert_and_its_visibility"],
        },
        "meta": facts["meta"]}
    return facts


# ============================================================================== decisions
BLOCKER_METRICS = {"oot_gini_floor"}
BLOCKER_TESTS = {"leakage"}
REPLACEMENT_TESTS = ("out_of_time", "fairness", "explanation_stability")


def _flagged(rows):
    return [r for r in rows if r["status"] in ("warn", "fail")]


def decide(R: Results) -> dict:
    """Rule-based verdicts. Documented in the report; never produced by the LLM."""
    out = {}
    for role, name in R.roles.items():
        rows = R.headline(name)
        fails = [r["metric"] for r in rows if r["status"] == "fail"]
        warns = [r["metric"] for r in rows if r["status"] == "warn"]
        blockers = [r["metric"] for r in rows if r["status"] == "fail"
                    and (r["test"] in BLOCKER_TESTS or r["metric"] in BLOCKER_METRICS)]
        verdict = ("Not approved" if blockers else "Approved with conditions" if fails
                   else "Approved with monitoring" if warns else "Approved")
        out[role] = {"model": name, "verdict": verdict, "headline_fail_count": len(fails),
                     "headline_warn_count": len(warns), "blocking_findings": blockers}

    champ, chall = R.roles["champion"], R.roles["challenger"]
    gain = (R.d(chall, "oot_gini")["gini"] - R.d(champ, "oot_gini")["gini"])
    flagged = lambda n: {r["metric"] for r in _flagged(R.headline(n)) if r["test"] in REPLACEMENT_TESTS}
    new_findings = sorted(flagged(chall) - flagged(champ))
    reasons = []
    if gain < MATERIAL_GINI_GAIN:
        reasons.append(f"Gini gain of {gain:.3f} is below the materiality threshold of {MATERIAL_GINI_GAIN}")
    reasons += [f"new finding not present for the champion: `{m}`" for m in new_findings]
    out["replacement"] = {"challenger_recommended_to_replace_champion": not reasons, "reasons": reasons}
    return out


CONDITION_RULES = [
    ({"oot_calibration_ratio", "oot_bands_underpredicted", "valid_calibration_ratio"},
     "Recalibrate the probabilities of default before use and re-estimate calibration at least quarterly, "
     "tracking observed against predicted default rates by origination vintage."),
    ({"csi_max_used_feature_train_to_oot", "csi_max_used_feature_train_to_valid"},
     "Re-base the stability reference population on a recent matured sample, and review inputs whose "
     "availability changed between vintages (see the stability section)."),
    ({"oot_group_calibration_dev_emp_length", "oot_group_calibration_dev_income_quartile",
      "oot_group_calibration_dev_addr_state", "oot_group_calibration_dev_home_ownership",
      "oot_group_calibration_dev_verification_status"},
     "Investigate the groups with the largest calibration deviation and document a justification or a remediation."),
    ({"geographic_importance_share"},
     "Obtain legal and compliance review of geographic inputs, or remove them, before any deployment."),
    ({"gini_drop_train_to_oot", "gini_drop_valid_to_oot", "calibration_slope", "oot_calibration_slope",
      "valid_calibration_slope"},
     "Reduce overfitting (stronger regularisation or simpler structure) and re-test calibration slope."),
    ({"importance_rank_corr_champion_vs_challenger_oot", "importance_rank_corr_train_to_oot",
      "importance_rank_corr_train_to_valid"},
     "Document why the challenger relies on different drivers than the champion before it is considered for replacement."),
    ({"vintage_gini_loss"},
     "Remove or re-derive inputs whose availability changed between vintages."),
]


def conditions(R: Results) -> list[dict]:
    flagged = {}
    for role, name in R.roles.items():
        for r in _flagged(R.headline(name)):
            flagged.setdefault(r["metric"], set()).add(role)
    out = []
    for metrics, text in CONDITION_RULES:
        hit = sorted(m for m in metrics if m in flagged)
        if hit:
            roles = sorted({x for m in hit for x in flagged[m]})
            out.append({"condition": text, "applies_to": roles, "triggered_by": hit})
    out.append({"condition": "Add leading-indicator monitors (origination volume growth, mix shifts) because "
                             "outcome-based calibration monitoring only alerts after the outcome lag.",
                "applies_to": ["champion", "challenger"], "triggered_by": ["monitoring simulation"]})
    return out


# ============================================================================== templates
def _n(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def _s(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _move(x):
    return f"falls by {abs(x)}%" if x < 0 else f"rises by {abs(x)}%"


def _t_discrimination(f):
    d = f["discrimination"]; c, x = d["champion"], d["challenger"]
    return (f"The champion, a {f['meta']['champion']['description']}, reaches a Gini of {c['gini_oot']} and a KS of "
            f"{c['ks_oot']} on the out-of-time sample, against {c['gini_train']} in training and {c['gini_valid']} in "
            f"validation; its Gini {_move(c['gini_change_train_to_oot_pct'])} between training and out-of-time. "
            f"The challenger, a {f['meta']['challenger']['description']}, reaches {x['gini_oot']} out-of-time, a gain of "
            f"{d['challenger_gini_gain_oot']} over the champion, but its Gini {_move(x['gini_change_train_to_oot_pct'])} "
            f"between training and out-of-time. The materiality threshold for replacing the champion is a gain of "
            f"{d['materiality_threshold_gini_gain']}.")


def _t_calibration(f):
    d = f["calibration"]; c, x = d["champion"], d["challenger"]
    rc, rx = c["ratio_after_recalibrating_on_validation_year"], x["ratio_after_recalibrating_on_validation_year"]
    return (f"On the out-of-time sample the champion predicts a mean default rate of {c['oot_mean_predicted_pd_pct']}% "
            f"against an observed {c['oot_observed_default_rate_pct']}%, a calibration ratio of "
            f"{c['oot_calibration_ratio']} (status: {c['oot_calibration_ratio_status']}); the challenger predicts "
            f"{x['oot_mean_predicted_pd_pct']}%, a ratio of {x['oot_calibration_ratio']} (status: "
            f"{x['oot_calibration_ratio_status']}). {c['risk_bands_underpredicted']} of {c['risk_bands_evaluated']} risk "
            f"bands are significantly under-predicted for the champion and {x['risk_bands_underpredicted']} for the "
            f"challenger. The calibration slope is {c['oot_calibration_slope']} for the champion and "
            f"{x['oot_calibration_slope']} for the challenger. Recalibrating on the validation year improves the "
            f"champion's ratio only to {rc['intercept']} and the challenger's to {rx['intercept']}.")


def _t_stability(f):
    d = f["stability"]; c, x = d["champion"], d["challenger"]
    text = (f"The score distribution is stable for the champion, with a PSI of {c['score_psi_oot']} out-of-time and "
            f"{c['score_psi_monitor']} in the monitoring year. The default rate rose from {c['train_default_rate_pct']}% in "
            f"training to {c['oot_default_rate_pct']}% out-of-time. The worst input drift for the champion is a CSI of "
            f"{c['worst_input_csi_oot']} on {c['worst_input_csi_feature']}; {c['model_inputs_with_csi_above_amber']} of "
            f"{c['model_inputs_total']} inputs are above the amber threshold. ")
    if c["n_inputs_with_changed_availability"]:
        k = c["n_inputs_with_changed_availability"]
        text += (f"{_s(k, 'input')} changed availability between vintages; removing {'it' if k == 1 else 'them'} "
                 f"changes the champion's out-of-time Gini to {c['gini_oot_without_those_inputs']} "
                 f"({c['relative_gini_loss_without_those_inputs_pct']}% relative loss). ")
    text += (f"No post-origination field is used by the champion ({c['post_origination_inputs_in_model']} found); the "
             f"strongest single input has a Gini of {c['strongest_single_input_gini']}, whereas a deliberately leaky "
             f"model reaches {c['deliberately_leaky_model_gini_oot']} against {c['clean_model_gini_oot']} for the clean model.")
    return text


def _t_explain(f):
    d = f["explainability_fairness"]; c, x = d["champion"], d["challenger"]
    groups = "; ".join(f"{k} ({v['worst_group']}: {v['deviation_pct']}%, {v['status']})"
                       for k, v in c["group_calibration"].items())
    text = (f"The champion's driver ranking is stable over time, with a rank correlation of "
            f"{c['importance_rank_correlation_train_to_oot']} between training and out-of-time and "
            f"{c['top5_driver_overlap_train_to_oot']} of the top 5 drivers shared. ")
    if x["importance_rank_correlation_vs_champion"] is not None:
        text += (f"The challenger's drivers agree only partly with the champion's: a rank correlation of "
                 f"{x['importance_rank_correlation_vs_champion']} and {x['top5_driver_overlap_vs_champion']} of the top 5 "
                 f"drivers shared. ")
    text += (f"Group calibration deviation from the portfolio for the champion: {groups}. Geographic inputs carry "
             f"{c['geographic_importance_share_pct']}% of the champion's importance and "
             f"{x['geographic_importance_share_pct']}% of the challenger's. The data contains no protected "
             f"attributes, so these are proxy checks only.")
    return text


def _alert(d):
    return "never alerted" if d["first_alert_vintage"] == "never" else f"first alerted on the {d['first_alert_vintage']} vintage"


def _t_monitoring(f):
    m = f["monitoring"]["champion"]; det, pf, rc = m["detection"], m["portfolio"], m["recalibration_lagged"]
    cal, psi, gini = det["calibration"], det["score_psi"], det["gini"]
    text = (f"For the champion, the score PSI {_alert(psi)} and the Gini monitor {_alert(gini)}. The calibration monitor "
            f"first raised an alert on the {cal['first_alert_vintage']} "
            f"vintage, visible from {cal['alert_visible_from']} because outcomes lag by "
            f"{f['meta']['monitoring_outcome_lag_months']} months, and first turned red on the {cal['first_red_vintage']} "
            f"vintage. Between the first deteriorating vintage and the moment the alert became visible, "
            f"{_n(pf['loans_issued_between_first_calibration_alert_and_its_visibility'])} loans were issued. The mean predicted "
            f"PD moved from {pf['mean_predicted_pd_first_month_pct']}% to {pf['mean_predicted_pd_last_outcome_month_pct']}% "
            f"while the observed default rate moved from {pf['observed_default_rate_first_month_pct']}% to "
            f"{pf['observed_default_rate_last_outcome_month_pct']}%, and monthly volume grew by a factor of "
            f"{pf['volume_growth_multiple']}.")
    if rc.get("months_compared"):
        text += (f" A recalibration using only outcomes already known reduces the calibration ratio in {rc['last_month']} from "
                 f"{rc['last_ratio_raw']} to {rc['last_ratio_recalibrated']}, which narrows the gap without closing it.")
    return text


def _t_executive(f):
    v = f["verdicts"]; k = f["executive_summary"]["key_numbers"]
    chall_rec = v["replacement"]["challenger_recommended_to_replace_champion"]
    return (f"The champion ({f['meta']['champion']['description']}) receives the verdict '{v['champion']['verdict']}' and the "
            f"challenger ({f['meta']['challenger']['description']}) '{v['challenger']['verdict']}'. The challenger is "
            f"{'' if chall_rec else 'not '}recommended to replace the champion. Out-of-time Gini is "
            f"{k['champion_gini_oot']} for the champion and {k['challenger_gini_oot']} for the challenger, but calibration "
            f"is the main weakness: the observed to predicted default ratio is {k['champion_calibration_ratio_oot']} and "
            f"{k['challenger_calibration_ratio_oot']} respectively, while the champion's score PSI is only "
            f"{k['champion_score_psi_oot']}. The conditions listed below are required before use.")


TEMPLATES = {"executive_summary": _t_executive, "discrimination": _t_discrimination,
             "calibration": _t_calibration, "stability": _t_stability,
             "explainability_fairness": _t_explain, "monitoring": _t_monitoring}

SECTION_GUIDE = {
    "executive_summary": "State both verdicts and the replacement decision exactly as given, then the two or three "
                         "most important numbers: out-of-time Gini, calibration ratios, and the contrast between "
                         "the score PSI and the calibration alert.",
    "discrimination": "Compare champion and challenger on Gini, KS and the change from training to out-of-time; "
                      "say whether the gain passes the materiality threshold.",
    "calibration": "Explain predicted versus observed default rates, the calibration ratio and its status, "
                   "under-predicted risk bands, the slope, and what recalibration achieves.",
    "stability": "Cover score PSI, input drift, inputs whose availability changed and the effect of removing them, "
                 "and the leakage checks including the deliberately leaky model.",
    "explainability_fairness": "Cover driver stability, agreement between the models, group calibration deviations "
                               "(name the worst groups), geographic inputs, and the proxy-only limitation.",
    "monitoring": "Explain which monitors alerted and when, the outcome lag, the loans issued before the alert was "
                  "visible, the predicted-versus-observed divergence and what lagged recalibration achieves.",
}

SYSTEM_PROMPT = (
    "You are a model-validation analyst writing ONE section of an independent validation report for a bank's "
    "model-risk committee. Rules: use ONLY the facts provided; quote every number exactly as it appears in the facts; "
    "never compute, estimate, round differently or invent numbers; write years in full (2012-2013, not 2012-13); "
    "do not use headings, tables, bullet or numbered lists; write 2 short paragraphs of plain prose, 110 to 190 words "
    "in total; neutral, evidence-based tone; do not change or add verdicts or approval decisions; mention a "
    "limitation if the facts contain one.")


# =============================================================================== sections
@dataclass
class SectionResult:
    key: str
    text: str
    source: str                       # "llm" | "template"
    attempts: int = 0
    rejected: list = field(default_factory=list)
    prompt_sha256: str | None = None


def _user_prompt(key: str, facts_subset: dict) -> str:
    return (f"Section to write: {TITLES[key]}.\nWhat to cover: {SECTION_GUIDE[key]}\n\nFACTS (JSON):\n"
            + json.dumps(facts_subset, indent=1, default=str))


def write_section(key: str, facts: dict, llm, max_attempts: int = 3) -> SectionResult:
    subset = facts[key] if key != "executive_summary" else facts["executive_summary"]
    subset = {"meta": facts["meta"], **subset} if "meta" not in subset else subset
    fallback = TEMPLATES[key](facts)

    if llm is None:
        return SectionResult(key, fallback, "template")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _user_prompt(key, subset)}]
    sha = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    rejected = []
    for attempt in range(1, max_attempts + 1):
        draft = llm.chat(messages, temperature=0.0).strip()
        verdict = check_text(draft, subset)
        if verdict.ok and draft:
            return SectionResult(key, draft, "llm", attempt, rejected, sha)
        rejected.append({"attempt": attempt, "violations": verdict.violations})
        messages = messages + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": "Your draft contains numbers that are not in the facts: "
             + ", ".join(v["number"] for v in verdict.violations)
             + ". Rewrite the section using only numbers that appear in the facts, copied exactly."}]
    return SectionResult(key, fallback, "template", max_attempts, rejected, sha)


# ================================================================================ render
def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        if float(v).is_integer() or abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def make_figures(val: dict, mon: dict, out_dir: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    R = Results(val)
    champ = R.roles["champion"]
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {}

    dec = R.d(champ, "oot_calibration_ratio").get("deciles")
    if dec:
        d = pd.DataFrame(dec)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        lim = float(max(d["mean_pd"].max(), d["observed_dr"].max(), d["ci_high"].max())) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="gray", label="perfect calibration")
        ax.errorbar(d["mean_pd"], d["observed_dr"], yerr=[d["observed_dr"] - d["ci_low"], d["ci_high"] - d["observed_dr"]],
                    fmt="o", color="#1f77b4", capsize=3, label="observed (95% interval)")
        ax.set_xlabel("predicted PD (decile mean)"); ax.set_ylabel("observed default rate")
        ax.set_title("Champion: calibration by decile, out-of-time (2015)"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_dir / "calibration_deciles.png", dpi=130); plt.close(fig)
        figs["calibration_deciles"] = "figures/calibration_deciles.png"

    t = pd.DataFrame(mon["models"][champ]["monthly"])
    t["month"] = pd.to_datetime(t["month"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(t["month"], t["mean_pd"], label="predicted (mean PD)", color="#1f77b4")
    ax.plot(t["month"], pd.to_numeric(t["observed_default_rate"], errors="coerce"), label="observed", color="#d62728")
    ax.set_title("Champion: predicted vs observed default rate"); ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax = axes[1]
    ax.plot(t["month"], pd.to_numeric(t["calibration_ratio_roll3"], errors="coerce"), label="raw model", color="#1f77b4")
    ax.plot(t["month"], pd.to_numeric(t["calibration_ratio_recal_roll3"], errors="coerce"),
            label="lagged recalibration", color="#2ca02c")
    for y, c in [(1.0, "gray"), (1.1, "orange"), (1.2, "red")]:
        ax.axhline(y, ls="--", color=c, lw=1)
    ax.set_title("Calibration ratio (observed / predicted)"); ax.legend(frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out_dir / "monitoring.png", dpi=130); plt.close(fig)
    figs["monitoring"] = "figures/monitoring.png"
    return figs


def render_markdown(val: dict, mon: dict, facts: dict, sections: dict[str, SectionResult],
                    figures: dict[str, str], llm_info: dict) -> str:
    R = Results(val)
    v = facts["verdicts"]
    findings = []
    for role in ("champion", "challenger"):
        name = R.roles[role]
        for r in sorted(_flagged(R.headline(name)), key=lambda r: (r["status"] != "fail", r["test"])):
            findings.append([f"{name} ({role})", r["test"], r["metric"], _fmt(r["value"]), r["status"]])
    conds = conditions(R)

    out = [f"# Independent model validation report: FinRisk Lab credit-risk models",
           f"*Validation run `{val['run_id']}` generated {val['generated_at']}. Report generated "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.*", "",
           "> Demonstration on public LendingClub data. Not a real credit decision. "
           "Verdicts and conditions are rule-based; prose was written under a number-verification guard (Appendix B).", "",
           "## 1. Executive summary", sections["executive_summary"].text, "",
           md_table(["Model", "Role", "Verdict", "Headline fails", "Headline warns"],
                    [[v[r]["model"], r, f"**{v[r]['verdict']}**", v[r]["headline_fail_count"], v[r]["headline_warn_count"]]
                     for r in ("champion", "challenger")]), "",
           f"**Replacement decision:** the challenger is "
           f"{'' if v['replacement']['challenger_recommended_to_replace_champion'] else '**not** '}recommended to replace "
           f"the champion." + ("" if v["replacement"]["challenger_recommended_to_replace_champion"]
                               else " Reasons: " + "; ".join(v["replacement"]["reasons"]) + "."), "",
           "## 2. Scope and data",
           f"Public LendingClub accepted loans (Kaggle), {facts['meta']['loan_term_months']}-month loans only. Split by "
           f"origination year: training {facts['meta']['train_years']}, validation {facts['meta']['validation_year']}, "
           f"out-of-time test {facts['meta']['out_of_time_year']}, monitoring {facts['meta']['monitoring_year']} "
           "(outcomes censored). Target: charged off or defaulted versus fully paid; unresolved loans are excluded.", "",
           f"- Champion: `{facts['meta']['champion']['model']}` (registry version {facts['meta']['champion']['version']}), "
           f"{facts['meta']['champion']['description']}",
           f"- Challenger: `{facts['meta']['challenger']['model']}` (registry version {facts['meta']['challenger']['version']}), "
           f"{facts['meta']['challenger']['description']}", "",
           "## 3. Discriminatory power", sections["discrimination"].text, ""]
    d = facts["discrimination"]
    out += [md_table(["Model", "Gini train", "Gini valid", "Gini out-of-time", "KS out-of-time"],
                     [[role, d[role]["gini_train"], d[role]["gini_valid"], d[role]["gini_oot"], d[role]["ks_oot"]]
                      for role in ("champion", "challenger")]), "",
            "## 4. Calibration", sections["calibration"].text, ""]
    if "calibration_deciles" in figures:
        out += [f"![Calibration by decile]({figures['calibration_deciles']})", ""]
    c = facts["calibration"]
    out += [md_table(["Model", "Predicted PD %", "Observed %", "Ratio", "Slope", "Bands under-predicted"],
                     [[role, c[role]["oot_mean_predicted_pd_pct"], c[role]["oot_observed_default_rate_pct"],
                       c[role]["oot_calibration_ratio"], c[role]["oot_calibration_slope"],
                       f"{c[role]['risk_bands_underpredicted']} of {c[role]['risk_bands_evaluated']}"]
                      for role in ("champion", "challenger")]), "",
            "## 5. Stability, sensitivity and leakage", sections["stability"].text, "",
            "## 6. Explainability and fairness", sections["explainability_fairness"].text, "",
            "## 7. Monitoring", sections["monitoring"].text, ""]
    if "monitoring" in figures:
        out += [f"![Monitoring]({figures['monitoring']})", ""]
    out += ["## 8. Findings and conditions", "### Findings (warn / fail, headline tests)",
            md_table(["Model", "Test", "Metric", "Value", "Status"], findings) if findings else "None.", "",
            "### Conditions for use (rule-based)"]
    out += [f"{i}. {c['condition']} *(applies to: {', '.join(c['applies_to'])}; triggered by: "
            f"{', '.join(c['triggered_by'])})*" for i, c in enumerate(conds, 1)]
    out += ["", "## 9. Limitations",
            "- Fairness checks are proxy analyses: the data contains no race, gender or age.",
            "- Outcomes of 2016 originations are censored, so outcome-based monitoring covers 2014-2015 only.",
            "- The validation policy thresholds (Appendix A) are analyst choices, not regulatory limits.",
            "- Results come from one public dataset with one temporal split; they do not establish how the models "
            "would behave at another lender or in another cycle.",
            "- The number guard verifies figures, not interpretation; a human reviewer should read the prose.", "",
            "## Appendix A. Validation policy (thresholds)",
            md_table(["Check", "Warn", "Fail"],
                     [[k, x["warn"], x["fail"]] for k, x in val["thresholds"].items()]), "",
            "## Appendix B. How this report was generated",
            f"- Verdicts, conditions, tables and figures: deterministic code from `validation_results.json` and "
            f"`monitoring_results.json`.",
            f"- Prose: {llm_info['description']}", "",
            md_table(["Section", "Written by", "Attempts", "Numbers rejected before acceptance"],
                     [[TITLES[k], s.source, s.attempts, sum(len(r["violations"]) for r in s.rejected)]
                      for k, s in sections.items()]), ""]
    return "\n".join(out)


_CSS = ("body{font-family:Georgia,serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}"
        "table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}th,td{border:1px solid #bbb;padding:4px 8px}"
        "th{background:#f0f0f0}blockquote{border-left:4px solid #bbb;margin:1rem 0;padding:.2rem 1rem;color:#555}"
        "img{max-width:100%}code{background:#f4f4f4;padding:0 3px}@media print{body{margin:0}}")


def render_html(md: str, base_dir: Path) -> str | None:
    try:
        import markdown
    except ImportError:
        return None
    body = markdown.markdown(md, extensions=["tables"])

    def inline(m):
        p = base_dir / m.group(1)
        if not p.exists():
            return m.group(0)
        return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'

    body = re.sub(r'src="(figures/[^"]+)"', inline, body)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Validation report</title><style>{_CSS}</style></head><body>{body}</body></html>"


# ==================================================================================== CLI
def generate(reports_dir: Path, llm=None, max_attempts: int = 3, with_figures: bool = True) -> dict:
    val, mon = load_inputs(reports_dir)
    facts = build_facts(val, mon)
    sections = {k: write_section(k, facts, llm, max_attempts) for k in LLM_SECTIONS}
    figures = make_figures(val, mon, reports_dir / "figures") if with_figures else {}
    if llm is None:
        info = {"description": "deterministic templates (no LLM was used); every number comes straight from the results files.",
                "provider": None, "model": None}
    else:
        info = {"description": f"written by a language model (Groq, model `{getattr(llm, 'model', 'unknown')}`, temperature 0) "
                               "from a facts dictionary; each draft was checked by the number guard and rejected drafts "
                               "were retried, then replaced by a deterministic template if they kept failing.",
                "provider": "groq", "model": getattr(llm, "model", None)}
    md = render_markdown(val, mon, facts, sections, figures, info)
    (reports_dir / "validation_report.md").write_text(md, encoding="utf-8")
    html = render_html(md, reports_dir)
    if html:
        (reports_dir / "validation_report.html").write_text(html, encoding="utf-8")
    audit = {"generated_at": datetime.now().isoformat(timespec="seconds"), "llm": info,
             "guard": "every number and date in LLM prose must be supported by the section's facts",
             "sections": [{"section": k, "source": s.source, "attempts": s.attempts, "prompt_sha256": s.prompt_sha256,
                           "rejected": s.rejected} for k, s in sections.items()]}
    (reports_dir / "validation_report_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"markdown": md, "html": html, "sections": sections, "facts": facts, "audit": audit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="deterministic templates only")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    llm = None
    if not args.no_llm:
        from finrisk.genai.groq_client import GroqLLM, LLMError
        try:
            llm = GroqLLM(model=args.model)
        except LLMError as exc:
            raise SystemExit(f"{exc}\n(or run with --no-llm for a report without LLM prose)")
    reports_dir = get_settings().reports_dir
    try:
        res = generate(reports_dir, llm, args.max_attempts)
    except Exception as exc:
        from finrisk.genai.groq_client import LLMError
        if isinstance(exc, LLMError):
            raise SystemExit(str(exc))
        raise
    print(f"Wrote {reports_dir / 'validation_report.md'}")
    if res["html"]:
        print(f"Wrote {reports_dir / 'validation_report.html'}  (open in a browser, Print -> Save as PDF)")
    for s in res["sections"].values():
        extra = f", {sum(len(r['violations']) for r in s.rejected)} numbers rejected" if s.rejected else ""
        print(f"  {TITLES[s.key]:<38} {s.source:<9} attempts={s.attempts}{extra}")
    if llm is not None:
        print(f"LLM calls: {llm.calls} (cache hits: {llm.cache_hits})")


if __name__ == "__main__":
    main()
