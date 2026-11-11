-- 03_features.sql
-- staging.model_sample (loans + split) -> features.loan_features
--
-- Only information known at origination is used. Post-origination columns
-- (payments, recoveries, ...) are deliberately NOT carried over.
-- grade / sub_grade / int_rate are LendingClub's own underwriting outputs:
-- they are kept here but trained with and without (see features/build.py).

DROP TABLE IF EXISTS features.loan_features CASCADE;

CREATE TABLE features.loan_features AS
SELECT
    m.loan_id,
    m.split,
    m.issue_d,
    m.issue_year,
    m.is_default,

    -- application
    m.loan_amnt,
    m.annual_inc,
    m.dti,
    m.emp_length_years,
    m.home_ownership,
    m.verification_status,
    m.purpose,
    m.addr_state,
    m.initial_list_status,

    -- credit bureau
    ((m.fico_range_low + m.fico_range_high) / 2.0)::double precision        AS fico_mid,
    m.delinq_2yrs,
    m.inq_last_6mths,
    m.mths_since_last_delinq,
    m.mths_since_last_record,
    m.open_acc,
    m.pub_rec,
    m.revol_bal,
    m.revol_util,
    m.total_acc,
    m.mort_acc,
    m.pub_rec_bankruptcies,
    m.tot_cur_bal,
    m.total_rev_hi_lim,
    m.avg_cur_bal,
    m.bc_util,
    m.num_bc_sats,

    -- engineered
    ((m.issue_d - m.earliest_cr_line_d) / 365.25)::double precision          AS credit_history_years,
    CASE WHEN m.annual_inc > 0 THEN m.loan_amnt / m.annual_inc END           AS loan_to_income,
    CASE WHEN m.annual_inc > 0 THEN m.revol_bal / m.annual_inc END           AS revol_bal_to_income,

    -- lender-assigned (LendingClub's own risk assessment)
    m.grade,
    m.sub_grade,
    m.int_rate
FROM staging.model_sample m;

ALTER TABLE features.loan_features ADD PRIMARY KEY (loan_id);
CREATE INDEX idx_loan_features_split ON features.loan_features (split);
ANALYZE features.loan_features;
