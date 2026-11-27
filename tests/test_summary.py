"""README results block, CV bullets and README helpers."""
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
    assert "mermaid" in text and "Ethical & Methodological Disclosures" in text
