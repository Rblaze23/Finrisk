"""Unit tests for the Agentic Credit Analyst, Regulatory RAG, and PDF Generator."""
import json
from pathlib import Path

import pytest

from finrisk.agent.credit_analyst import (
    ARCHETYPES,
    compute_counterfactual_path,
    copilot_answer,
    evaluate_underwriting_policy,
    generate_credit_memo,
)
from finrisk.genai.rag import RegulatoryRAG, answer_regulatory_question, query_regulations
from finrisk.models.export import PortableScorecard
from finrisk.reports.pdf_generator import (
    generate_underwriting_memo_pdf,
    generate_validation_report_pdf,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


@pytest.fixture(scope="module")
def scorecard():
    p = REPORTS_DIR / "champion_scorecard.json"
    if not p.exists():
        pytest.skip("champion_scorecard.json not found")
    return PortableScorecard.load(p)


def test_archetypes_structure():
    assert len(ARCHETYPES) >= 4
    for key, arch in ARCHETYPES.items():
        assert "name" in arch
        assert "inputs" in arch
        assert "fico_score" in arch["inputs"]
        assert "dti" in arch["inputs"]
        assert "loan_amnt" in arch["inputs"]


def test_underwriting_policy_evaluation():
    # Prime case
    prime_inputs = ARCHETYPES["alex_chen_prime"]["inputs"]
    eval_prime = evaluate_underwriting_policy(prime_inputs, 0.035, "R1")
    assert eval_prime.verdict == "APPROVED"
    assert eval_prime.passed is True
    assert len(eval_prime.strengths) >= 2

    # Distressed case (hard cutoffs)
    distressed_inputs = ARCHETYPES["taylor_smith_distressed"]["inputs"]
    eval_dist = evaluate_underwriting_policy(distressed_inputs, 0.28, "R6")
    assert eval_dist.verdict == "DECLINED"
    assert eval_dist.passed is False
    assert len(eval_dist.exceptions) >= 1


def test_counterfactual_optimization(scorecard):
    inputs = ARCHETYPES["jordan_taylor_near_prime"]["inputs"]
    cf = compute_counterfactual_path(scorecard, inputs)
    assert cf.current_band in ("R2", "R3", "R4")
    assert len(cf.actions) >= 1
    assert cf.target_pd <= cf.current_pd


def test_generate_credit_memo(scorecard):
    inputs = ARCHETYPES["alex_chen_prime"]["inputs"]
    memo = generate_credit_memo(scorecard, inputs, "Alex Chen")
    assert memo.applicant_name == "Alex Chen"
    assert memo.verdict == "APPROVED"
    assert memo.risk_band == "R1"
    assert len(memo.reasons) > 0
    assert "Credit Underwriting & Advisory Memo" in memo.memo_markdown


def test_copilot_answers(scorecard):
    inputs = ARCHETYPES["jordan_taylor_near_prime"]["inputs"]
    memo = generate_credit_memo(scorecard, inputs, "Jordan Taylor")

    ans_why = copilot_answer("Why was this applicant scored this way?", memo, inputs)
    assert len(ans_why) > 20
    assert memo.risk_band in ans_why

    ans_cf = copilot_answer("How can this borrower improve their score?", memo, inputs)
    assert len(ans_cf) > 20


def test_regulatory_rag_retrieval():
    rag = RegulatoryRAG()
    hits = rag.retrieve("outcomes analysis model validation", top_k=2)
    assert len(hits) >= 1
    assert "SR 11-7" in hits[0].standard or "EBA" in hits[0].standard
    assert hits[0].citation is not None

    fcra_hits = rag.retrieve("adverse action notice FCRA", top_k=1)
    assert len(fcra_hits) >= 1
    assert "FCRA" in fcra_hits[0].standard


def test_pdf_generators():
    val_path = REPORTS_DIR / "validation_results.json"
    mon_path = REPORTS_DIR / "monitoring_results.json"
    if not val_path.exists() or not mon_path.exists():
        pytest.skip("Reports not found")

    val = json.loads(val_path.read_text(encoding="utf-8"))
    mon = json.loads(mon_path.read_text(encoding="utf-8"))

    # Test Validation Report PDF
    val_pdf = generate_validation_report_pdf(val, mon)
    assert isinstance(val_pdf, bytes)
    assert val_pdf.startswith(b"%PDF-1.4")
    assert len(val_pdf) > 1000

    # Test Underwriting Memo PDF
    sc_path = REPORTS_DIR / "champion_scorecard.json"
    sc = PortableScorecard.load(sc_path)
    inputs = ARCHETYPES["jordan_taylor_near_prime"]["inputs"]
    memo = generate_credit_memo(sc, inputs, "Jordan Taylor")
    memo_pdf = generate_underwriting_memo_pdf(memo, inputs)
    assert isinstance(memo_pdf, bytes)
    assert memo_pdf.startswith(b"%PDF-1.4")
    assert len(memo_pdf) > 1000
