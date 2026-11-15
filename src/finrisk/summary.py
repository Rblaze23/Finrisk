"""README results block and CV bullets, generated from the results files.

    python -m finrisk.summary --readme          # refresh the block between the RESULTS markers
    python -m finrisk.summary --cv              # print CV bullets with your real numbers
    python -m finrisk.summary --demo-url URL    # set the live-demo link in the README

Every figure comes from reports/validation_results.json and reports/monitoring_results.json
(through the same facts dictionary the validation report uses), so the README cannot drift
away from the results.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from finrisk.config import PROJECT_ROOT, get_settings
from finrisk.genai.report_writer import Results, _move, build_facts, load_inputs

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
DEMO_START, DEMO_END = "<!-- DEMO:START -->", "<!-- DEMO:END -->"
SHORT = {"scorecard": "scorecard", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def f3(v) -> str:
    """Fixed three decimals, so 1.19 prints as 1.190."""
    return f"{v:.3f}"


def github_slug(url: str) -> str | None:
    """'user/repo' from an https or ssh GitHub remote URL."""
    m = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def collect(val: dict, mon: dict) -> dict:
    """Flat dictionary of every number the README and the CV bullets use."""
    facts = build_facts(val, mon)
    R = Results(val)
    c, x = "champion", "challenger"
    kinds = {r: val["models"][R.roles[r]]["kind"] for r in (c, x)}
    d, cal, st, ef = (facts[k] for k in ("discrimination", "calibration", "stability", "explainability_fairness"))
    mo = facts["monitoring"][c]
    det, pf, rec = mo["detection"], mo["portfolio"], mo["recalibration_lagged"]
    v = facts["verdicts"]

    groups = [(attr, g["worst_group"], g["deviation_pct"], g["status"])
              for attr, g in ef[c]["group_calibration"].items() if g["deviation_pct"] is not None]
    worst = max(groups, key=lambda t: t[2]) if groups else (None, None, None, None)
    sizes = {s: R.d(R.roles[c], f"{s}_gini").get("n") for s in ("train", "valid", "oot")}
    graded = [r for r in R.headline(R.roles[c]) if r["status"] in ("pass", "warn", "fail")]

    return {
        "facts": facts, "names": {c: SHORT[kinds[c]], x: SHORT[kinds[x]]},
        "model_ids": {c: R.roles[c], x: R.roles[x]},
        "sizes": sizes, "n_loans": sum(n or 0 for n in sizes.values()), "n_checks": len(graded),
        "gini": {c: d[c]["gini_oot"], x: d[x]["gini_oot"]}, "gini_train": {c: d[c]["gini_train"], x: d[x]["gini_train"]},
        "ks": {c: d[c]["ks_oot"], x: d[x]["ks_oot"]},
        "gini_change": {c: d[c]["gini_change_train_to_oot_pct"], x: d[x]["gini_change_train_to_oot_pct"]},
        "gain": d["challenger_gini_gain_oot"],
        "ratio": {c: cal[c]["oot_calibration_ratio"], x: cal[x]["oot_calibration_ratio"]},
        "pred_pd": {c: cal[c]["oot_mean_predicted_pd_pct"], x: cal[x]["oot_mean_predicted_pd_pct"]},
        "obs": cal[c]["oot_observed_default_rate_pct"],
        "bands": {c: f"{cal[c]['risk_bands_underpredicted']} of {cal[c]['risk_bands_evaluated']}",
                  x: f"{cal[x]['risk_bands_underpredicted']} of {cal[x]['risk_bands_evaluated']}"},
        "slope": {c: cal[c]["oot_calibration_slope"], x: cal[x]["oot_calibration_slope"]},
        "static_recal": cal[c]["ratio_after_recalibrating_on_validation_year"]["intercept"],
        "verdict": {c: v[c]["verdict"], x: v[x]["verdict"]},
        "replace": v["replacement"]["challenger_recommended_to_replace_champion"],
        "replace_reasons": v["replacement"]["reasons"],
        "psi_never": det["score_psi"]["first_alert_vintage"] == "never",
        "psi_months": det["score_psi"]["months_evaluated"], "psi_first": det["score_psi"]["first_alert_vintage"],
        "cal_first": det["calibration"]["first_alert_vintage"], "cal_visible": det["calibration"]["alert_visible_from"],
        "cal_first_red": det["calibration"]["first_red_vintage"],
        "lag": facts["meta"]["monitoring_outcome_lag_months"],
        "loans_before_alert": pf["loans_issued_between_first_calibration_alert_and_its_visibility"],
        "pd_first": pf["mean_predicted_pd_first_month_pct"], "pd_last": pf["mean_predicted_pd_last_outcome_month_pct"],
        "obs_first": pf["observed_default_rate_first_month_pct"], "obs_last": pf["observed_default_rate_last_outcome_month_pct"],
        "vol_multiple": pf["volume_growth_multiple"], "first_month": pf["first_month"], "last_month": pf["last_month_with_outcomes"],
        "recal_raw": rec.get("last_ratio_raw"), "recal_after": rec.get("last_ratio_recalibrated"),
        "recal_month": rec.get("last_month"),
        "worst_group": worst, "geo": {c: ef[c]["geographic_importance_share_pct"], x: ef[x]["geographic_importance_share_pct"]},
        "driver_corr": ef[x]["importance_rank_correlation_vs_champion"],
        "leaky": st[c]["deliberately_leaky_model_gini_oot"], "clean": st[c]["clean_model_gini_oot"],
        "leak_found": st[c]["post_origination_inputs_in_model"],
        "vintage_n": st[c]["n_inputs_with_changed_availability"],
        "vintage_loss": st[c]["relative_gini_loss_without_those_inputs_pct"],
        "train_dr": st[c]["train_default_rate_pct"], "oot_dr": st[c]["oot_default_rate_pct"],
    }


def results_block(K: dict) -> str:
    c, x = "champion", "challenger"
    n = K["names"]
    lines = [
        "### Key results",
        f"*Generated from `reports/*.json`. Out-of-time = loans issued in 2015; models are trained on 2012-2013. "
        f"{(format(K['n_loans'], ',') + ' resolved') if K['n_loans'] else 'Resolved'} 36-month loans.*", "",
        f"| | {n[c]} (champion) | {n[x]} (challenger) |", "|---|---|---|",
        f"| Gini, out-of-time | {f3(K['gini'][c])} | {f3(K['gini'][x])} |",
        f"| KS, out-of-time | {f3(K['ks'][c])} | {f3(K['ks'][x])} |",
        f"| Gini, training | {f3(K['gini_train'][c])} | {f3(K['gini_train'][x])} |",
        f"| Calibration ratio, out-of-time (observed / predicted) | {f3(K['ratio'][c])} | {f3(K['ratio'][x])} |",
        f"| Predicted vs observed default rate | {K['pred_pd'][c]}% vs {K['obs']}% | {K['pred_pd'][x]}% vs {K['obs']}% |",
        f"| Risk bands under-predicted | {K['bands'][c]} | {K['bands'][x]} |",
        f"| Calibration slope (1.0 = ideal) | {f3(K['slope'][c])} | {f3(K['slope'][x])} |",
        f"| Validation verdict (rule-based) | **{K['verdict'][c]}** | **{K['verdict'][x]}** |", "",
        "**What the validation found**", "",
        f"1. **Ranking power is stable, probabilities are not.** Out-of-time Gini is {f3(K['gini'][c])} ({n[c]}) and "
        f"{f3(K['gini'][x])} ({n[x]}), yet both models under-predict 2015 defaults: calibration ratios of "
        f"{f3(K['ratio'][c])} and {f3(K['ratio'][x])} (observed {K['obs']}% against {K['pred_pd'][c]}% / "
        f"{K['pred_pd'][x]}% predicted).",
    ]
    if K["psi_never"]:
        lines.append(
            f"2. **Input-drift monitoring alone would not have caught it.** Over {K['psi_months']} monthly vintages the score PSI "
            f"never alerted. The calibration monitor first flagged the {K['cal_first']} vintage, but because outcomes lag by "
            f"{K['lag']} months it was visible only from {K['cal_visible']}, after {K['loans_before_alert']:,} further loans "
            f"had been issued.")
    else:
        lines.append(
            f"2. **Monitors disagree on timing.** The score PSI first alerted on the {K['psi_first']} vintage; the calibration "
            f"monitor flagged {K['cal_first']} but was visible only from {K['cal_visible']} because of a {K['lag']}-month "
            f"outcome lag ({K['loans_before_alert']:,} loans issued in between).")
    lines.append(
        f"3. **The model saw improving risk while reality worsened.** Between {K['first_month']} and {K['last_month']} the mean "
        f"predicted PD went from {K['pd_first']}% to {K['pd_last']}% while the observed default rate went from "
        f"{K['obs_first']}% to {K['obs_last']}%, and monthly origination volume grew by a factor of {K['vol_multiple']}.")
    verdict = ("is recommended to replace the champion" if K["replace"]
               else "is **not** recommended to replace the champion")
    lines.append(
        f"4. **More complex is not automatically better.** {n[x]} adds {f3(K['gain'])} Gini, but its Gini "
        f"{_move(K['gini_change'][x])} from training to out-of-time (champion: {_move(K['gini_change'][c])}), "
        f"and it {verdict}"
        + (": " + "; ".join(K["replace_reasons"]) if K["replace_reasons"] else "") + ".")
    if K["recal_raw"] is not None:
        lines.append(
            f"5. **Recalibration helps only partly.** A recalibration fitted on 2014 leaves a ratio of {f3(K['static_recal'])}; "
            f"recalibrating monthly with only outcomes already known moves the ratio in {K['recal_month']} from "
            f"{f3(K['recal_raw'])} to {f3(K['recal_after'])}.")
    attr, grp, dev, status = K["worst_group"]
    if attr:
        lines.append(
            f"6. **Proxy fairness checks.** The largest group calibration deviation is {dev}% ({attr}: {grp}, {status}); "
            f"geographic inputs carry {K['geo'][c]}% of the champion's importance and {K['geo'][x]}% of the challenger's. "
            f"The data has no protected attributes, so these are proxy checks only.")
    if K["leaky"] is not None:
        lines.append(
            f"7. **Leakage is easy to create and easy to detect.** Adding post-origination fields lifts the out-of-time Gini "
            f"from {f3(K['clean'])} to {f3(K['leaky'])}; the production models use none ({K['leak_found']} found).")
    lines += ["", "![Calibration by decile](reports/figures/calibration_deciles.png)", "",
              "![Monitoring simulation](reports/figures/monitoring.png)"]
    return "\n".join(lines)


def cv_bullets(K: dict, n_tests: int | None = None) -> list[str]:
    c, x = "champion", "challenger"
    n = K["names"]
    out = [
        f"Built FinRisk Lab, an end-to-end credit-risk modelling and validation platform on "
        f"{(format(K['n_loans'], ',') + ' ') if K['n_loans'] else ''}LendingClub loans "
        f"(Python, PostgreSQL, MLflow, FastAPI, Streamlit, Docker): WoE scorecard champion vs {n[x]} challenger, trained on "
        f"2012-2013 and tested out-of-time on 2015.",
        f"Found that both models under-predicted 2015 defaults (calibration ratio {f3(K['ratio'][c])} and {f3(K['ratio'][x])}) while the "
        f"Gini held at {f3(K['gini'][c])}; showed that recalibrating on the prior year still left a ratio of {f3(K['static_recal'])}, "
        f"so periodic recalibration is required.",
    ]
    if K["psi_never"]:
        out.append(
            f"Simulated {K['psi_months']} months of portfolio monitoring: score PSI never alerted, while the calibration monitor "
            f"flagged the {K['cal_first']} vintage but became visible only {K['lag']} months later, after "
            f"{K['loans_before_alert']:,} loans had been issued; proposed leading-indicator monitors.")
    else:
        out.append(
            f"Simulated {K['psi_months']} months of portfolio monitoring and measured alert timing: calibration drift flagged on "
            f"the {K['cal_first']} vintage but visible only {K['lag']} months later ({K['loans_before_alert']:,} loans issued).")
    rec = (f"recommended against replacing the scorecard with {n[x]} (+{f3(K['gain'])} Gini; {len(K['replace_reasons'])} "
           f"blocking reason{'s' if len(K['replace_reasons']) != 1 else ''})" if not K["replace"]
           else f"recommended replacing the scorecard with {n[x]} (+{f3(K['gain'])} Gini)")
    out.append(
        f"Designed {K['n_checks']} graded validation checks per model (calibration, PSI/CSI, leakage, vintage sensitivity, "
        f"explanation stability, fairness proxies) with a pass/warn/fail policy and rule-based verdicts; {rec}.")
    out.append(
        "Built a guarded GenAI report writer (Groq): the LLM drafts prose only, a number guard rejects any figure not found in "
        "the results files and retries, and every attempt is logged in an audit trail.")
    tests = f", covered by {n_tests} automated tests and CI" if n_tests else ", covered by automated tests and CI"
    out.append(f"Shipped a FastAPI scoring service with SHAP-based reason codes and a Streamlit dashboard{tests}.")
    return out


def update_readme(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    i, j = text.find(START), text.find(END)
    if i < 0 or j < i:
        raise SystemExit(f"{path} has no {START} ... {END} markers.")
    path.write_text(text[: i + len(START)] + "\n" + block.strip() + "\n" + text[j:], encoding="utf-8")


def set_demo_url(path: Path, url: str) -> None:
    text = path.read_text(encoding="utf-8")
    i, j = text.find(DEMO_START), text.find(DEMO_END)
    if i < 0 or j < i:
        raise SystemExit(f"{path} has no {DEMO_START} ... {DEMO_END} markers.")
    line = f"**Live demo:** [{url}]({url})"
    path.write_text(text[: i + len(DEMO_START)] + "\n" + line + "\n" + text[j:], encoding="utf-8")


def count_tests(root: Path) -> int | None:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=root,
                             capture_output=True, text=True, timeout=180).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--readme", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--demo-url", default=None)
    args = ap.parse_args()
    if not (args.readme or args.cv or args.demo_url):
        args.readme = args.cv = True

    readme = PROJECT_ROOT / "README.md"
    if args.demo_url:
        set_demo_url(readme, args.demo_url)
        print(f"Set the live demo link in {readme.name}")
    if args.readme or args.cv:
        val, mon = load_inputs(get_settings().reports_dir)
        K = collect(val, mon)
        if args.readme:
            update_readme(readme, results_block(K))
            print(f"Updated the results block in {readme.name}")
        if args.cv:
            print("\nCV BULLETS (copy into your Finance / Banking AI & Risk CV)\n" + "-" * 60)
            print("FinRisk Lab - credit-risk modelling, validation and monitoring (personal project)")
            for b in cv_bullets(K, count_tests(PROJECT_ROOT)):
                print("- " + b)
            print("\nTech: Python, SQL, PostgreSQL, scikit-learn, XGBoost, LightGBM, SHAP, MLflow, FastAPI, "
                  "Streamlit, Docker, Groq (LLM), pytest, GitHub Actions")
            print("\nKeep a claim only if it is true for you: run the report with Groq at least once before keeping the "
                  "Groq bullet, and keep 'CI' only once the GitHub Actions badge is green.")


if __name__ == "__main__":
    main()
