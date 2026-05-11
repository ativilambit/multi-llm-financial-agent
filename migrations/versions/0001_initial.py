"""Initial run metadata schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("earnings_date", sa.Text(), nullable=True),
        sa.Column("run_environment", sa.Text(), nullable=True),
        sa.Column("started_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("iterative", sa.Boolean(), nullable=True),
        sa.Column("iterations_completed", sa.Integer(), nullable=True),
        sa.Column("config_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("synthesis_path", sa.Text(), nullable=True),
        sa.Column("synthesizer_provider", sa.Text(), nullable=True),
        sa.Column("synthesizer_model", sa.Text(), nullable=True),
        sa.Column("verifier_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("drive_folder_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at_utc",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at_utc",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_runs_symbol", "runs", ["symbol"])
    op.create_index("ix_runs_earnings_date", "runs", ["earnings_date"])
    op.create_index("ix_runs_started_at_utc", "runs", ["started_at_utc"])

    op.create_table(
        "provider_responses",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "run_id",
            sa.Text(),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("latency_s", sa.Numeric(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=True),
        sa.Column("web_search_enabled", sa.Boolean(), nullable=True),
        sa.Column("succeeded", sa.Boolean(), nullable=True),
        sa.Column("error_kind", sa.Text(), nullable=True),
        sa.Column("response_path", sa.Text(), nullable=True),
    )

    op.create_table(
        "outcomes",
        sa.Column(
            "run_id",
            sa.Text(),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "recorded_at_utc",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("earnings_day_open", sa.Numeric(), nullable=True),
        sa.Column("earnings_day_high", sa.Numeric(), nullable=True),
        sa.Column("earnings_day_low", sa.Numeric(), nullable=True),
        sa.Column("earnings_day_close", sa.Numeric(), nullable=True),
        sa.Column("next_trading_day_open", sa.Numeric(), nullable=True),
        sa.Column("next_trading_day_close", sa.Numeric(), nullable=True),
        sa.Column("one_week_later_close", sa.Numeric(), nullable=True),
        sa.Column("direction_vs_prior_close", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), server_default=sa.text("'manual'"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "direction_vs_prior_close IN ('up','down','flat')",
            name="outcomes_direction_vs_prior_close_check",
        ),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "run_id",
            sa.Text(),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("horizon", sa.Text(), nullable=True),
        sa.Column("predicted_probability_up", sa.Numeric(), nullable=True),
        sa.Column("predicted_range_low", sa.Numeric(), nullable=True),
        sa.Column("predicted_range_high", sa.Numeric(), nullable=True),
        sa.Column("predicted_point", sa.Numeric(), nullable=True),
        sa.Column("source", sa.Text(), server_default=sa.text("'manual'"), nullable=False),
        sa.Column(
            "created_at_utc",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("outcomes")
    op.drop_table("provider_responses")
    op.drop_index("ix_runs_started_at_utc", table_name="runs")
    op.drop_index("ix_runs_earnings_date", table_name="runs")
    op.drop_index("ix_runs_symbol", table_name="runs")
    op.drop_table("runs")

