"""Column catalogue for the LendingClub accepted-loans file.

Single source of truth for which columns are loaded and what they mean for
modelling. The leakage tests (validation/leakage.py) use POST_ORIGINATION_COLUMNS.
"""

ID_COLUMN = "id"
STATUS_COLUMN = "loan_status"

# --- Known at application / origination time --------------------------------
NUMERIC_ORIGINATION = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_util", "num_bc_sats",
]

CATEGORICAL_ORIGINATION = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "zip_code", "addr_state",
    "initial_list_status", "application_type",
]

DATE_ORIGINATION = ["issue_d", "earliest_cr_line"]

ORIGINATION_COLUMNS = NUMERIC_ORIGINATION + CATEGORICAL_ORIGINATION + DATE_ORIGINATION

# Outputs of LendingClub's own underwriting model. A bank building an
# independent PD model would not have these, so Phase 2 trains with and without.
LENDER_ASSIGNED_COLUMNS = ["grade", "sub_grade", "int_rate"]

# --- Only known AFTER the loan was issued (leakage candidates) --------------
POST_ORIGINATION_COLUMNS = [
    "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag",
]

# Everything loaded into raw.loans (order preserved, no duplicates)
RAW_COLUMNS = list(
    dict.fromkeys(
        [ID_COLUMN, STATUS_COLUMN] + ORIGINATION_COLUMNS + POST_ORIGINATION_COLUMNS
    )
)
