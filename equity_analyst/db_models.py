from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    earnings_date: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_environment: Mapped[str | None] = mapped_column(Text, nullable=True)
    env: Mapped[str] = mapped_column(String(16), nullable=False, server_default="production")

    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    iterative: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    iterations_completed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    synthesis_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    synthesizer_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    synthesizer_model: Mapped[str | None] = mapped_column(Text, nullable=True)

    verifier_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    drive_folder_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    provider_responses: Mapped[list[ProviderResponseRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    outcome: Mapped[OutcomeRow | None] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    predictions: Mapped[list[PredictionRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_runs_symbol", "symbol"),
        Index("ix_runs_earnings_date", "earnings_date"),
        Index("ix_runs_started_at_utc", "started_at_utc"),
        Index("ix_runs_env", "env"),
        CheckConstraint("env IN ('production','test')", name="ck_runs_env_values"),
    )


class ProviderResponseRow(Base):
    __tablename__ = "provider_responses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)

    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)

    latency_s: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    web_search_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_kind: Mapped[str | None] = mapped_column(Text, nullable=True)

    response_path: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[RunRow] = relationship(back_populates="provider_responses")


class OutcomeRow(Base):
    __tablename__ = "outcomes"

    run_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    recorded_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    earnings_day_open: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    earnings_day_high: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    earnings_day_low: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    earnings_day_close: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    next_trading_day_open: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    next_trading_day_close: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    one_week_later_close: Mapped[float | None] = mapped_column(Numeric, nullable=True)

    direction_vs_prior_close: Mapped[str | None] = mapped_column(String, nullable=True)

    source: Mapped[str] = mapped_column(Text, server_default="manual", nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[RunRow] = relationship(back_populates="outcome")

    __table_args__ = (
        CheckConstraint(
            "direction_vs_prior_close IN ('up','down','flat')",
            name="ck_outcomes_direction_vs_prior_close",
        ),
    )


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    horizon: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_probability_up: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    predicted_range_low: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    predicted_range_high: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    predicted_point: Mapped[float | None] = mapped_column(Numeric, nullable=True)

    source: Mapped[str] = mapped_column(Text, server_default="manual", nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped[RunRow] = relationship(back_populates="predictions")

