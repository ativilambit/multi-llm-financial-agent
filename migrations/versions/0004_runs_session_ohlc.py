"""Add runs.session_* columns for post-close regular-session OHLC lock.

Revision ID: 0004_runs_session_ohlc
Revises: 0003_runs_run_document
Create Date: 2026-05-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_runs_session_ohlc"
down_revision = "0003_runs_run_document"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("session_trade_date", sa.Date(), nullable=True))
    op.add_column(
        "runs",
        sa.Column("session_open", sa.Numeric(18, 4), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("session_high", sa.Numeric(18, 4), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("session_low", sa.Numeric(18, 4), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("session_close", sa.Numeric(18, 4), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column(
            "session_partial",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "runs",
        sa.Column("session_snapshot_at_utc", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("runs", sa.Column("session_source", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "session_source")
    op.drop_column("runs", "session_snapshot_at_utc")
    op.drop_column("runs", "session_partial")
    op.drop_column("runs", "session_close")
    op.drop_column("runs", "session_low")
    op.drop_column("runs", "session_high")
    op.drop_column("runs", "session_open")
    op.drop_column("runs", "session_trade_date")
