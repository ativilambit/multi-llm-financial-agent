"""Add runs.run_document JSONB (nullable full run.json snapshot).

Revision ID: 0003_runs_run_document
Revises: 0002_add_runs_env
Create Date: 2026-05-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_runs_run_document"
down_revision = "0002_add_runs_env"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "run_document",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("runs", "run_document")
