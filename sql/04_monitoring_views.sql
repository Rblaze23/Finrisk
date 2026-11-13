-- 04_monitoring_views.sql
-- Extends monitoring.monthly_portfolio and adds reporting views.
-- Idempotent: safe to run repeatedly (the simulation runs it before writing).

ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS share_high_risk               DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS csi_max_used                  DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS csi_worst_feature             TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS calibration_ratio_roll3       DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS gini_roll3                    DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS mean_pd_recal                 DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS calibration_ratio_recal_roll3 DOUBLE PRECISION;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_score_psi              TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_csi                    TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_calibration            TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS status_gini                   TEXT;
ALTER TABLE monitoring.monthly_portfolio ADD COLUMN IF NOT EXISTS outcomes_available            BOOLEAN;

-- Observed default rate by issue month and split (outcomes of recent vintages are censored)
CREATE OR REPLACE VIEW monitoring.v_default_rate_by_vintage AS
SELECT date_trunc('month', issue_d)::date            AS month,
       split,
       COUNT(*)                                      AS n_loans,
       SUM(is_default)                               AS n_defaults,
       ROUND(AVG(is_default::numeric), 4)            AS default_rate
FROM staging.model_sample
GROUP BY 1, 2;

-- Share of loans in each risk band per issue month (score band migration)
CREATE OR REPLACE VIEW monitoring.v_band_mix_by_month AS
WITH agg AS (
    SELECT date_trunc('month', c.issue_d)::date      AS month,
           p.model_name,
           p.risk_band,
           COUNT(*)                                  AS n_loans
    FROM results.predictions p
    JOIN staging.loans_clean c USING (loan_id)
    GROUP BY 1, 2, 3
)
SELECT month, model_name, risk_band, n_loans,
       ROUND(100.0 * n_loans / SUM(n_loans) OVER (PARTITION BY month, model_name), 2) AS pct_of_month
FROM agg;

-- Most recent traffic lights per model
CREATE OR REPLACE VIEW monitoring.v_latest_status AS
SELECT DISTINCT ON (model_name)
       model_name, month, n_loans, mean_pd, score_psi, csi_max_used,
       status_score_psi, status_csi, status_calibration, status_gini
FROM monitoring.monthly_portfolio
ORDER BY model_name, month DESC;
