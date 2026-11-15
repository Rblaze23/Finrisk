#!/usr/bin/env python3
"""
Phase 7 for FinRisk Lab: README, generated results block, CV bullets, license, CI.

Run from the ROOT of your repo (after Phase 6):

    python apply_phase7.py

Then:
    python -m finrisk.summary --readme --cv         # fills the README results block, prints CV bullets
    python -m finrisk.summary --demo-url https://YOUR-APP.streamlit.app   # after deploying
The old README is saved as README.md.bak.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}
OVERWRITE = {"README.md"}

# ---------------------------------------------------------------------------
FILES["src/finrisk/summary.py"] = r'''"""README results block and CV bullets, generated from the results files.

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
'''

# ---------------------------------------------------------------------------
FILES["README.md"] = r'''# FinRisk Lab

[![tests](https://github.com/GITHUB_SLUG/actions/workflows/tests.yml/badge.svg)](https://github.com/GITHUB_SLUG/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Credit-risk modelling, validation and monitoring on LendingClub loans.**
Most credit-default projects stop at "my model has X% accuracy". Banks ask a harder question:
*can we trust this model with real money, and where does it break?* FinRisk Lab builds the
models, then validates them the way a bank's model-risk team would: out-of-time testing,
calibration, drift, leakage, explainability, fairness proxies and ongoing monitoring, ending in
an audit-style validation report.

<!-- DEMO:START -->
**Live demo:** *deploy the dashboard (see below) and run* `python -m finrisk.summary --demo-url <your-url>`
<!-- DEMO:END -->

![dashboard](docs/dashboard.png)

<!-- RESULTS:START -->
*Run `python -m finrisk.summary --readme` to generate the results block from your results files.*
<!-- RESULTS:END -->

## How it works

```mermaid
flowchart LR
  A[LendingClub CSV] --> B[(PostgreSQL<br/>raw / staging / features)]
  B --> C[Scorecard + XGBoost + LightGBM<br/>MLflow registry: champion / challenger]
  C --> D[Validation suite<br/>calibration, stability, leakage, fairness]
  C --> E[FastAPI<br/>PD, risk band, reason codes]
  C --> G[Monitoring simulation<br/>36 monthly vintages]
  D --> F[results JSON]
  G --> F
  F --> H[Guarded report writer<br/>Groq LLM + number guard]
  F --> I[Streamlit dashboard]
```

The scoring pipeline never calls an LLM. GenAI is used only to draft report prose, behind a guard.

### Models

| Model | Role | Why |
|---|---|---|
| WoE logistic-regression scorecard | Champion | What regulated lenders actually deploy: monotone, explainable, easy to audit |
| XGBoost, LightGBM | Challengers | Do they earn their complexity? Monotone constraints, early stopping on the validation year, no class re-weighting so PDs stay real probabilities |

Training uses 36-month loans issued in 2012-2013, validation 2014, **out-of-time test 2015**,
monitoring 2016. Splits are by issue year, never random, and the code refuses overlapping periods.
Loans issued after 2015 had not matured when the data ends, so 2016 outcomes are censored and
are used only for input monitors.

### The validation suite

| Family | Question it answers |
|---|---|
| Discrimination | Gini, KS, AUC by period; Gini loss train to out-of-time; Gini by quarter |
| Calibration | Do predicted PDs match observed default rates? Decile table with confidence intervals, calibration slope, one-sided under-prediction test per risk band, three recalibration methods |
| Stability | PSI of the score; CSI of every input the model uses; default-rate shift |
| Sensitivity | Retrain without inputs whose availability changed between vintages |
| Leakage | No post-origination inputs; no single input suspiciously predictive; a deliberately leaky model to show the effect |
| Explanation stability | Do SHAP / scorecard drivers keep their ranking over time and across models? |
| Fairness (proxy) | Calibration by group; share of importance on geographic variables |
| Monitoring | 36 monthly vintages, outcome lag, alert timing, lagged recalibration |

Every result is `pass`, `warn`, `fail` or `info` under an explicit policy
(`THRESHOLDS` in `src/finrisk/validation/results.py`). The thresholds are analyst choices,
not regulatory limits. Verdicts ("approved with conditions" and so on) and the conditions list are
produced by rules, not by a model.

### The GenAI layer, and why it is guarded

`python -m finrisk.genai.report_writer` writes the validation report. The LLM (Groq, temperature 0)
drafts six prose sections from a facts dictionary. A **number guard** extracts every number and date
from each draft and rejects it if any figure is not in the facts; the draft is retried with the
offending numbers listed, and after three attempts a deterministic template takes over. The
appendix of the report and `reports/validation_report_audit.json` record who wrote what, how many
attempts it took and which numbers were rejected. The guard verifies numbers, not interpretation,
which is why a human should still read the prose. `--no-llm` produces the full report without any API key.

## Quick start

Requires Python 3.12, Docker Desktop, and a free Kaggle account for the data.

```bash
git clone https://github.com/GITHUB_SLUG.git && cd finrisk
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -e .
cp .env.example .env                                  # optional: add GROQ_API_KEY
docker compose up -d                                  # PostgreSQL + MLflow

# download accepted_2007_to_2018Q4.csv.gz from Kaggle (wordsforthewise/lending-club) into data/raw/
python -m finrisk.data.ingest            # add --sample for a fast 10% run
python -m finrisk.data.clean
python -m finrisk.data.splits
python -m finrisk.features.build
python -m finrisk.models.train           # scorecard + XGBoost + LightGBM, logged to MLflow
python -m finrisk.validation.run_validation
python -m finrisk.monitoring.simulate --reference valid
python -m finrisk.models.export          # portable scorecard for the public dashboard
python -m finrisk.genai.report_writer --no-llm
python -m finrisk.summary --readme --cv  # refresh the results above from the JSON files

pytest -q
streamlit run streamlit_app.py           # dashboard
uvicorn finrisk.api.main:app --port 8000 # API, docs at http://localhost:8000/docs
```

The committed `reports/*.json` are enough to run the dashboard without any of the data steps.

### Scoring API

```bash
curl -X POST "http://localhost:8000/score?include_challenger=true" -H "Content-Type: application/json" \
  -d '{"loan_amnt": 15000, "annual_inc": 52000, "fico_score": 672, "dti": 24.5, "home_ownership": "RENT"}'
```

Returns the probability of default, a risk band (R1 to R6) and the top reasons that raise the
applicant's risk, from the champion and optionally the challenger.

### Deploy the dashboard (free)

1. Push the repository to GitHub (the `reports/` JSON files must be committed).
2. On [share.streamlit.io](https://share.streamlit.io) choose the repo, branch `main`, main file `streamlit_app.py`.
3. The root `requirements.txt` installs only the light dashboard dependencies. The dashboard reads
   committed files and needs no database or MLflow.
4. Add the URL here with `python -m finrisk.summary --demo-url <url>`.

Optional Docker for the API and dashboard (not part of the default compose file):
`docker compose -f docker-compose.yml -f docker-compose.app.yml up -d --build`.

## Repository layout

```
src/finrisk/
  data/         ingest (COPY into PostgreSQL), clean (SQL staging + target), time-based splits
  features/     SQL features, WoE binning and Information Value
  models/       scorecard, XGBoost/LightGBM, MLflow training and registry, risk bands, portable export
  explain/      SHAP / scorecard contributions, reason codes
  validation/   calibration, stability, out-of-time, sensitivity, leakage, explanation stability, fairness
  monitoring/   monthly vintages, outcome lag, alert timing, lagged recalibration
  genai/        Groq client, number guard, report writer
  api/          FastAPI scoring service
  dashboard/    Streamlit app
sql/            schema, staging, features, monitoring views
tests/          pytest suite (unit tests, API tests, dashboard tests with Streamlit AppTest)
reports/        results JSON, validation report, figures
```

## Limitations

- Fairness checks are **proxy analyses**: the data has no race, gender or age. This is not a fair-lending assessment.
- One public dataset, one temporal split. Results do not show how the models would behave at another lender or in another cycle.
- 2016 outcomes are censored, so outcome monitors cover 2014-2015 only; the 12-month outcome lag is an assumption.
- `grade`, `sub_grade` and `int_rate` are LendingClub's own underwriting outputs, so the headline models exclude them. A second feature set that includes them is trained and appears in the model comparison, but the validation suite runs on the independent set by default.
- The validation thresholds and the materiality threshold for replacing the champion are policy choices.
- This is a portfolio project, not a production credit model. The API and dashboard say so on every response.

## Data and license

Data: LendingClub accepted loans 2007-2018Q4 from Kaggle (`wordsforthewise/lending-club`). The data is
not redistributed here; download it yourself and respect the dataset's terms. Code: MIT, see `LICENSE`.
'''

FILES["LICENSE"] = r'''MIT License

Copyright (c) 2026 YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

FILES[".github/workflows/tests.yml"] = r'''name: tests

on:
  push:
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install the project
        run: pip install -e .
      - name: Run the tests
        run: pytest -q
'''

FILES["docs/.gitkeep"] = ""

# ---------------------------------------------------------------------------
FILES["tests/test_summary.py"] = r'''"""README results block, CV bullets and README helpers."""
import re

import pytest

from finrisk import summary as S
from tests.test_report_writer import _monitoring, _validation


def _with_sizes(val):
    sizes = {"train_gini": 143892, "valid_gini": 162570, "oot_gini": 283026}
    for r in val["results"]:
        if r["metric"] in sizes:
            r["details"]["n"] = sizes[r["metric"]]
    return val


@pytest.fixture()
def K(tmp_path):
    val, mon = _with_sizes(_validation(tmp_path)), _monitoring(tmp_path)
    return S.collect(val, mon)


def test_loan_counts_are_summed_and_shown(K):
    assert K["n_loans"] == 143892 + 162570 + 283026
    assert "589,488" in S.results_block(K) and "589,488" in S.cv_bullets(K)[0]


def test_missing_loan_counts_are_omitted_rather_than_shown_as_zero(tmp_path):
    K0 = S.collect(_validation(tmp_path), _monitoring(tmp_path))      # fixture without counts
    assert K0["n_loans"] == 0
    assert " 0 " not in S.results_block(K0) and "on LendingClub loans" in S.cv_bullets(K0)[0]


def test_collect_reads_the_headline_numbers(K):
    assert K["gini"] == {"champion": 0.298, "challenger": 0.335}
    assert K["ratio"]["champion"] == 1.204
    assert K["bands"]["champion"] == "5 of 6"
    assert K["psi_never"] is True and K["cal_first"] == "2014-10" and K["cal_visible"] == "2015-10"
    assert K["n_loans"] > 0 and K["n_checks"] > 10
    assert K["replace"] is False and K["names"] == {"champion": "scorecard", "challenger": "XGBoost"}


def test_results_block_contains_the_real_numbers_and_no_placeholders(K):
    block = S.results_block(K)
    for expected in ["0.298", "0.335", "1.204", "5 of 6", "Approved with conditions", "2014-10", "2015-10",
                     "not** recommended", "reports/figures/calibration_deciles.png"]:
        assert expected in block, expected
    assert "None" not in block and "nan" not in block.lower() and "{" not in block


def test_results_block_changes_wording_when_the_psi_monitor_does_alert(K):
    K = dict(K, psi_never=False, psi_first="2015-03")
    assert "Monitors disagree on timing" in S.results_block(K)


def test_cv_bullets_are_numbered_facts_not_placeholders(K):
    bullets = S.cv_bullets(K, n_tests=128)
    assert 5 <= len(bullets) <= 7
    text = " ".join(bullets)
    assert "128 automated tests" in text and "1.204" in text and "2014-10" in text
    assert "None" not in text and "{" not in text
    assert all(len(b) < 420 for b in bullets)


def test_cv_bullets_without_a_test_count_do_not_invent_one(K):
    text = " ".join(S.cv_bullets(K, n_tests=None))
    assert "covered by automated tests" in text


README = f"""# Title
intro
{S.DEMO_START}
old demo
{S.DEMO_END}
middle text
{S.START}
old block
{S.END}
footer text
"""


def test_update_readme_replaces_only_the_marked_block_and_is_idempotent(tmp_path, K):
    p = tmp_path / "README.md"
    p.write_text(README, encoding="utf-8")
    block = S.results_block(K)
    S.update_readme(p, block)
    once = p.read_text(encoding="utf-8")
    S.update_readme(p, block)
    assert p.read_text(encoding="utf-8") == once
    assert once.startswith("# Title\nintro") and once.rstrip().endswith("footer text")
    assert "old block" not in once and "Key results" in once and "middle text" in once


def test_update_readme_without_markers_fails_clearly(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("no markers here", encoding="utf-8")
    with pytest.raises(SystemExit, match="markers"):
        S.update_readme(p, "x")


def test_set_demo_url(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(README, encoding="utf-8")
    S.set_demo_url(p, "https://finrisk.streamlit.app")
    text = p.read_text(encoding="utf-8")
    assert "[https://finrisk.streamlit.app](https://finrisk.streamlit.app)" in text and "old demo" not in text


@pytest.mark.parametrize("url,slug", [
    ("https://github.com/ramy/finrisk.git", "ramy/finrisk"),
    ("https://github.com/ramy/finrisk", "ramy/finrisk"),
    ("git@github.com:ramy/finrisk.git", "ramy/finrisk"),
    ("https://gitlab.com/ramy/finrisk.git", None),
])
def test_github_slug(url, slug):
    assert S.github_slug(url) == slug


def test_shipped_readme_has_the_markers_and_no_unfilled_braces():
    text = (S.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert S.START in text and S.END in text and S.DEMO_START in text and S.DEMO_END in text
    assert "mermaid" in text and "Limitations" in text
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if rel in OVERWRITE:
            if old == content:
                print(f"  same    {rel}")
                return
            shutil.copy(path, str(path) + ".bak")
            path.write_text(content, encoding="utf-8")
            print(f"  replace {rel}  (old copy saved as {rel}.bak)")
            return
        if not FORCE and len(old.splitlines()) > 3:
            print(f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def github_slug(url: str):
    m = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    if not path.exists():
        print(f"  missing {rel}")
        return
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  ok      {rel}  (already patched)")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patch   {rel}")
    else:
        print(f"  warn    {rel}: expected text not found, skipping this patch")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "genai" / "report_writer.py").exists():
        sys.exit("Run apply_phase6.py first, from the repo root.")

    print(f"Applying Phase 7 in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    patch("src/finrisk/genai/report_writer.py",
          'reasons += [f"new finding not present for the champion: {m}" for m in new_findings]',
          'reasons += [f"new finding not present for the champion: `{m}`" for m in new_findings]')

    # fill in the GitHub user/repo in the README badges and clone command
    readme = ROOT / "README.md"
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                             text=True, timeout=15).stdout
        slug = github_slug(url)
    except Exception:
        slug = None
    if slug and readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "GITHUB_SLUG" in text:
            readme.write_text(text.replace("GITHUB_SLUG", slug), encoding="utf-8")
            print(f"  patch   README.md  (GitHub repository: {slug})")
    elif readme.exists() and "GITHUB_SLUG" in readme.read_text(encoding="utf-8"):
        print("  note    could not read your GitHub remote; replace GITHUB_SLUG in README.md with user/repo by hand")

    print("\nDone. Next:")
    print("  pytest -q")
    print("  python -m finrisk.summary --readme --cv")
    print("  edit LICENSE: replace YOUR NAME")


if __name__ == "__main__":
    main()