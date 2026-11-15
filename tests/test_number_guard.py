"""Unit tests for the number guard."""
import pytest

from finrisk.genai.number_guard import check_text, collect_allowed, extract_numbers

FACTS = {
    "gini_oot": 0.298, "ratio": 1.204, "pct": 14.9, "fraction": 0.149, "loans": 250990, "volume": 11495,
    "periods": "2012-2013", "year": 2015, "first_alert": "2014-10", "n": 5, "of": 6,
    "term": 36, "change": -17.0, "psi": 0.012,
}


def ok(text):
    return check_text(text, FACTS)


def test_exact_and_rounded_numbers_pass():
    assert ok("The Gini is 0.298 and the ratio 1.204; rounded, 1.20 and 0.3.").ok


def test_percent_forms_match_percentage_or_fraction_facts():
    assert ok("The rate was 14.9% (also 14.9 percent).").ok
    assert check_text("Observed 14.9%.", {"x": 0.149}).ok        # fraction stored, percent written


def test_thousands_separators_and_integers():
    assert ok("About 250,990 loans were issued, with 11,495 in the first month.").ok


def test_invented_number_is_rejected_and_reported_with_context():
    r = ok("The Gini is 0.35 on the test sample.")
    assert not r.ok
    assert r.violations[0]["number"] == "0.35"
    assert "Gini" in r.violations[0]["context"]


def test_over_precise_number_is_rejected():
    assert not ok("The ratio is 1.2043.").ok          # fact only has 1.204


def test_rounding_to_a_nicer_figure_is_rejected_for_integers():
    assert not ok("Roughly 251,000 loans were issued.").ok
    assert not ok("About 250,000 loans.").ok


def test_wrong_percent_is_rejected():
    assert not ok("The default rate was 15.9%.").ok


def test_sign_is_ignored_so_a_drop_can_be_described_in_words():
    assert ok("The Gini fell by 17.0%.").ok


def test_dates_must_appear_in_the_facts():
    assert ok("The first alert came on the 2014-10 vintage.").ok
    r = ok("The first alert came on the 2014-11 vintage.")
    assert not r.ok and r.violations[0]["number"] == "2014-11"


def test_year_ranges_and_single_years():
    assert ok("Training used 2012-2013 and the test year was 2015.").ok
    assert not ok("Training used 2012-13.").ok         # abbreviated range introduces 13


def test_reference_values_zero_one_hundred_are_allowed():
    assert ok("A ratio of 1.00 is perfect and 100% would be certain.").ok


def test_tokens_glued_to_letters_are_not_numbers():
    assert ok("Bands R5 and R6 and quartile Q1 were affected.").ok


def test_list_markers_are_ignored():
    assert ok("1. The Gini is 0.298.\n2) The ratio is 1.204.").ok


def test_36_month_phrase_uses_the_term_fact():
    assert ok("The 36-month loans were used.").ok
    assert not check_text("The 60-month loans were used.", FACTS).ok


def test_banker_and_half_up_rounding_are_both_accepted():
    assert check_text("The value is 0.13.", {"v": 0.125}).ok      # half up
    assert check_text("The value is 0.12.", {"v": 0.125}).ok      # python round


def test_collect_allowed_reads_numbers_inside_strings_and_lists():
    values, dates = collect_allowed({"a": ["x 12.5 y", 7], "d": "vintage 2014-10", "b": True, "n": None})
    assert 12.5 in values and 7.0 in values and 2014.0 not in values
    assert "2014-10" in dates


def test_extract_numbers_basic():
    nums, dates = extract_numbers("Gini 0.298, 14.9% and 1,234 loans on 2014-10.")
    assert [round(n.value, 3) for n in nums] == [0.298, 14.9, 1234.0]
    assert nums[1].percent and not nums[0].percent
    assert dates[0][0] == "2014-10"
