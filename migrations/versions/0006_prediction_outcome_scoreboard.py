"""Add views joining runs, outcomes, and predictions for scoring.

Revision ID: 0006_prediction_outcome_scoreboard
Revises: 0005_runs_synthesis_markdown
Create Date: 2026-05-14
"""

from __future__ import annotations

from alembic import op

revision = "0006_prediction_outcome_scoreboard"
down_revision = "0005_runs_synthesis_markdown"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE VIEW v_runs_outcomes_predictions AS
        SELECT
          r.run_id,
          r.symbol,
          r.earnings_date AS run_earnings_date,
          r.created_at_utc AS run_created_at_utc,
          o.recorded_at_utc AS outcome_recorded_at_utc,
          o.earnings_day_open,
          o.earnings_day_high,
          o.earnings_day_low,
          o.earnings_day_close,
          o.next_trading_day_open,
          o.next_trading_day_close,
          o.one_week_later_close,
          o.direction_vs_prior_close,
          o.source AS outcome_source,
          o.notes AS outcome_notes,
          p.id AS prediction_id,
          p.horizon AS prediction_horizon,
          p.predicted_probability_up,
          p.predicted_range_low,
          p.predicted_range_high,
          p.predicted_point,
          p.source AS prediction_source,
          p.created_at_utc AS prediction_created_at_utc
        FROM runs r
        LEFT JOIN outcomes o ON o.run_id = r.run_id
        LEFT JOIN predictions p ON p.run_id = r.run_id
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW prediction_outcome_scoreboard AS
        SELECT
          v.*,
          ha.horizon_actual,
          CASE
            WHEN v.predicted_point IS NOT NULL AND ha.horizon_actual IS NOT NULL
            THEN abs(v.predicted_point - ha.horizon_actual)
          END AS point_absolute_error
        FROM v_runs_outcomes_predictions v
        CROSS JOIN LATERAL (
          SELECT
            CASE v.prediction_horizon
              WHEN 'earnings_day_open' THEN v.earnings_day_open
              WHEN 'earnings_day_close' THEN v.earnings_day_close
              WHEN 'next_trading_day_open' THEN v.next_trading_day_open
              WHEN 'next_trading_day_close' THEN v.next_trading_day_close
              WHEN 'one_week_later_close' THEN v.one_week_later_close
              ELSE NULL::numeric
            END AS horizon_actual
        ) ha
        """
    )
    op.execute(
        """
        COMMENT ON VIEW v_runs_outcomes_predictions IS
        'Thin join: runs LEFT JOIN outcomes LEFT JOIN predictions. One row per '
        '(run, prediction) pair; prediction columns are NULL when no predictions exist yet. '
        'Multiple prediction rows per run are normal (one per horizon from prediction_extract).'
        """
    )
    op.execute(
        """
        COMMENT ON VIEW prediction_outcome_scoreboard IS
        'Like v_runs_outcomes_predictions plus horizon_actual (realized price for prediction_horizon) '
        'and point_absolute_error when predicted_point and horizon_actual are both non-null.'
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS prediction_outcome_scoreboard")
    op.execute("DROP VIEW IF EXISTS v_runs_outcomes_predictions")
