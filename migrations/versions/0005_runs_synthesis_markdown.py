"""Add runs.synthesis_markdown for DB-only prediction extraction in CI.

Revision ID: 0005_runs_synthesis_markdown
Revises: 0004_runs_session_ohlc
Create Date: 2026-05-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_runs_synthesis_markdown"
down_revision = "0004_runs_session_ohlc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("synthesis_markdown", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "synthesis_markdown")
