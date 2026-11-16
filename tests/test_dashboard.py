"""Runs the Streamlit app headlessly on small generated report files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from finrisk.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from finrisk.models.export import PortableScorecard
from finrisk.models.scorecard import Scorecard
from finrisk.monitoring.drift import detection_summary, monthly_monitoring, recalibration_summary
from finrisk.validation.calibration import calibration_table
from finrisk.validation.results import clean, make, write_json
from tests.test_export import _data
from tests.test_monitoring import _portfolio, _reference

APP = Path(__file__).resolve().parents[1] / "src" / "finrisk" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    d = tmp_path_factory.mktemp("reports")

    X, y = _data()
    sc = Scorecard(NUMERIC_FEATURES, CATEGORICAL_FEATURES, min_iv=0.005).fit(X, y)
    PortableScorecard.from_scorecard(sc, X.sample(2000, random_state=0), "1", "independent",
                                     n_train=len(X), train_default_rate=float(y.mean())).save(d / "champion_scorecard.json")

    p = sc.predict_proba(X)
    m = "scorecard_independent"
    rows = [
        make(m, "calibration", "oot_calibration_ratio", 1.2, "fail", 0.1, 0.2, "rule",
             {"deciles": calibration_table(y, p * 0.8)}),
        make(m, "calibration", "oot_brier", 0.12, "info"),
        make(m, "out_of_time", "train_gini", 0.30, "info"),
        make(m, "out_of_time", "valid_gini", 0.29, "info"),
        make(m, "out_of_time", "oot_gini", 0.30, "info"),
        make(m, "recalibration", "intercept_oot_calibration_ratio", 1.1, "warn", 0.1, 0.2, "rule",
             {"ratio_before": 1.2}),
        make(m, "stability", "score_psi_train_to_oot", 0.01, "pass", 0.1, 0.25, "rule"),
        make(m, "stability_detail", "csi_x_train_to_oot", 0.5, "fail", 0.1, 0.25, "rule"),
        make(m, "explanation_stability", "importance_rank_corr_train_to_oot", 0.97, "pass", 0.8, 0.6, "rule",
             {"top5_overlap": 0.8}),
    ]
    write_json(d / "validation_results.json", "val_test", rows,
               {m: {"role": "champion", "version": "1", "kind": "scorecard", "feature_set": "independent"}})

    df, pp = _portfolio(default_multiplier=1.3)
    ref, ref_p = _reference()
    t = monthly_monitoring(df, pp, ref, ref_p, ["f1"], ["c1"], pd.Timestamp("2015-12-01"))
    report = {"generated_at": "now",
              "params": {"feature_set": "independent", "outcome_lag_months": 12, "recal_window": 6,
                         "outcomes_known_until": "2015-12", "reference": "train"},
              "models": {m: {"role": "champion", "monthly": t, "detection": detection_summary(t),
                             "recalibration": recalibration_summary(t)}}}
    import json
    (d / "monitoring_results.json").write_text(json.dumps(clean(report)), encoding="utf-8")
    return d


PAGE_MAP = {
    "Agentic analyst": "🧑‍💼 Agentic Credit Analyst",
    "Score an applicant": "📊 Applicant Scoring & Sensitivity",
    "Validation results": "🛡️ Model Validation (SR 11-7)",
    "Explanation evaluation": "🔬 Explanation Evaluation",
    "Monitoring": "📈 Monitoring & Stress Testing",
    "GenAI": "🤖 Guarded GenAI & Regulatory RAG",
    "PDF reports": "📄 Executive PDF Reports",
    "About": "ℹ️ Architecture & Methodology",
}


def _app(reports, monkeypatch, page):
    monkeypatch.setenv("FINRISK_REPORTS_DIR", str(reports))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    full_page = PAGE_MAP.get(page, page)
    if full_page != "🧑‍💼 Agentic Credit Analyst":
        at.sidebar.radio[1].set_value(full_page).run()
    return at


def test_agentic_analyst_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Agentic analyst")
    assert not at.exception
    assert any("Agent Underwriting Verdict" in s.value for s in at.subheader)


def test_scoring_page_shows_pd_band_and_points(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Score an applicant")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Probability of default", "Risk band", "Scorecard points"]
    assert at.metric[0].value.endswith("%")
    assert at.metric[1].value in {"R1", "R2", "R3", "R4", "R5", "R6"}


def test_scoring_page_rejects_non_numeric_optional_fields(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Score an applicant")
    box = next(t for t in at.text_input if t.label == "Revolving balance ($)")
    box.set_value("lots")
    at.button[0].click().run()          # values inside a form only apply on submit
    assert not at.exception
    assert any("Not a number" in e.value for e in at.error)


def test_validation_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Validation results")
    assert not at.exception
    assert any("Validation Findings" in s.value for s in at.subheader)
    assert len(at.dataframe) >= 2


def test_explanation_evaluation_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Explanation evaluation")
    assert not at.exception
    assert any("Economic Monotonicity" in s.value for s in at.subheader)


def test_monitoring_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "Monitoring")
    assert not at.exception
    assert any("Detection Timeline" in s.value for s in at.subheader)


def test_pdf_reports_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "PDF reports")
    assert not at.exception
    assert any("Model Validation Report" in s.value for s in at.subheader)


def test_about_page_renders(reports, monkeypatch):
    at = _app(reports, monkeypatch, "About")
    assert not at.exception
    assert any("About FinRisk" in t.value for t in at.title)


def test_missing_report_files_give_a_helpful_warning_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("FINRISK_REPORTS_DIR", str(tmp_path))
    for page in ["Score an applicant", "Validation results", "Monitoring"]:
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        full_page = PAGE_MAP.get(page, page)
        if full_page != "🧑‍💼 Agentic Credit Analyst":
            at.sidebar.radio[1].set_value(full_page).run()
        assert not at.exception, page
        assert any("not found" in w.value for w in at.warning), page
