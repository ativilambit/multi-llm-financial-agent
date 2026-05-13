"""Add runs.env for deployment tier (production vs test).

Revision ID: 0002_add_runs_env
Revises: 0001_initial
Create Date: 2026-05-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_add_runs_env"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "env",
            sa.String(length=16),
            server_default=sa.text("'production'"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_runs_env_values",
        "runs",
        "env IN ('production','test')",
    )
    op.create_index("ix_runs_env", "runs", ["env"])


def downgrade() -> None:
    op.drop_index("ix_runs_env", table_name="runs")
    op.drop_constraint("ck_runs_env_values", "runs", type_="check")
    op.drop_column("runs", "env")
