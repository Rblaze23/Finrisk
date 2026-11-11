-- 02_staging.sql
-- raw.loans (all TEXT) -> staging.loans_clean (typed, with the default target).
--
-- Target definition:
--   is_default = 1 : loan_status contains 'Charged Off' or 'Default'
--   is_default = 0 : loan_status contains 'Fully Paid'
--   anything else (Current, Late, In Grace Period, ...) has no final outcome
--   yet and is EXCLUDED. This is documented in the validation report.
--
-- Re-running this script drops dependent objects (splits table and view).
-- Run `python -m finrisk.data.splits` afterwards.

-- ---------- safe parsing helpers ---------------------------------------------
CREATE OR REPLACE FUNCTION staging.to_num(t TEXT) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN c ~ '^-?[0-9]+([.][0-9]+)?$' THEN c::double precision END
    FROM (SELECT regexp_replace(t, '[^0-9.-]', '', 'g') AS c) s
$$;

-- 'Dec-2015' -> 2015-12-01
CREATE OR REPLACE FUNCTION staging.to_month(t TEXT) RETURNS DATE
LANGUAGE sql STABLE AS $$
    SELECT CASE WHEN t ~ '^[A-Za-z]{3}-[0-9]{4}$' THEN to_date(t, 'Mon-YYYY') END
$$;

-- ---------- staging table ------------------------------------------------------
DROP TABLE IF EXISTS staging.loans_clean CASCADE;

CREATE TABLE staging.loans_clean AS
WITH typed AS (
    SELECT
        r.id::bigint                                              AS loan_id,
        staging.to_month(r.issue_d)                               AS issue_d,
        EXTRACT(YEAR FROM staging.to_month(r.issue_d))::int       AS issue_year,
        staging.to_num(r.term)::int                               AS term_months,

        -- ===== known at origination =====
        staging.to_num(r.loan_amnt)                               AS loan_amnt,
        staging.to_num(r.funded_amnt)                             AS funded_amnt,
        staging.to_num(r.int_rate)                                AS int_rate,
        staging.to_num(r.installment)                             AS installment,
        r.grade,
        r.sub_grade,
        CASE
            WHEN r.emp_length IS NULL           THEN NULL
            WHEN r.emp_length LIKE '< 1%'       THEN 0
            WHEN r.emp_length LIKE '10+%'       THEN 10
            ELSE staging.to_num(r.emp_length)
        END                                                       AS emp_length_years,
        r.home_ownership,
        staging.to_num(r.annual_inc)                              AS annual_inc,
        r.verification_status,
        r.purpose,
        left(r.zip_code, 3)                                       AS zip3,
        r.addr_state,
        staging.to_num(r.dti)                                     AS dti,
        staging.to_num(r.delinq_2yrs)                             AS delinq_2yrs,
        staging.to_month(r.earliest_cr_line)                      AS earliest_cr_line_d,
        staging.to_num(r.fico_range_low)                          AS fico_range_low,
        staging.to_num(r.fico_range_high)                         AS fico_range_high,
        staging.to_num(r.inq_last_6mths)                          AS inq_last_6mths,
        staging.to_num(r.mths_since_last_delinq)                  AS mths_since_last_delinq,
        staging.to_num(r.mths_since_last_record)                  AS mths_since_last_record,
        staging.to_num(r.open_acc)                                AS open_acc,
        staging.to_num(r.pub_rec)                                 AS pub_rec,
        staging.to_num(r.revol_bal)                               AS revol_bal,
        staging.to_num(r.revol_util)                              AS revol_util,
        staging.to_num(r.total_acc)                               AS total_acc,
        r.initial_list_status,
        r.application_type,
        staging.to_num(r.mort_acc)                                AS mort_acc,
        staging.to_num(r.pub_rec_bankruptcies)                    AS pub_rec_bankruptcies,
        staging.to_num(r.collections_12_mths_ex_med)              AS collections_12_mths_ex_med,
        staging.to_num(r.acc_now_delinq)                          AS acc_now_delinq,
        staging.to_num(r.tot_coll_amt)                            AS tot_coll_amt,
        staging.to_num(r.tot_cur_bal)                             AS tot_cur_bal,
        staging.to_num(r.total_rev_hi_lim)                        AS total_rev_hi_lim,
        staging.to_num(r.avg_cur_bal)                             AS avg_cur_bal,
        staging.to_num(r.bc_util)                                 AS bc_util,
        staging.to_num(r.num_bc_sats)                             AS num_bc_sats,

        -- ===== only known AFTER issuance (leakage candidates) =====
        staging.to_num(r.out_prncp)                               AS out_prncp,
        staging.to_num(r.total_pymnt)                             AS total_pymnt,
        staging.to_num(r.total_rec_prncp)                         AS total_rec_prncp,
        staging.to_num(r.total_rec_int)                           AS total_rec_int,
        staging.to_num(r.total_rec_late_fee)                      AS total_rec_late_fee,
        staging.to_num(r.recoveries)                              AS recoveries,
        staging.to_num(r.collection_recovery_fee)                 AS collection_recovery_fee,
        staging.to_month(r.last_pymnt_d)                          AS last_pymnt_d,
        staging.to_num(r.last_pymnt_amnt)                         AS last_pymnt_amnt,
        staging.to_month(r.last_credit_pull_d)                    AS last_credit_pull_d,
        staging.to_num(r.last_fico_range_high)                    AS last_fico_range_high,
        staging.to_num(r.last_fico_range_low)                     AS last_fico_range_low,
        r.debt_settlement_flag,

        -- ===== target =====
        r.loan_status,
        CASE
            WHEN r.loan_status ~ 'Charged Off|Default' THEN 1
            WHEN r.loan_status ~ 'Fully Paid'          THEN 0
        END::smallint                                             AS is_default
    FROM raw.loans r
    WHERE r.id ~ '^[0-9]+$'
)
SELECT *
FROM typed
WHERE issue_d      IS NOT NULL
  AND term_months  IS NOT NULL
  AND is_default   IS NOT NULL;

ALTER TABLE staging.loans_clean ADD PRIMARY KEY (loan_id);
CREATE INDEX idx_loans_clean_issue ON staging.loans_clean (issue_year, term_months);
ANALYZE staging.loans_clean;
