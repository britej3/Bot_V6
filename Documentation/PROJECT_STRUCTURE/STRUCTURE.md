# Project Structure and Standards

This document defines the canonical folder layout, ownership, and boundaries. It also lists current discrepancies and a staged refactor plan (no breaking moves without approval).

## Canonical Layout

- `src/`
  - `api/` — FastAPI routers, models, dependencies
  - `core/` — core orchestration and long‑lived services
  - `config/` — typed configs and env wiring
  - `data_pipeline/` — WS managers, adapters, validators
  - `database/` — SQLAlchemy models, managers, DAO (embeddings)
  - `enhanced/` — optional perf/ML features (keep off hot path)
  - `learning/` — ML components (XGB ensemble, NN adapters, HPO)
  - `memory/` — MemoryService router and memory abstractions
  - `monitoring/` — metrics, health, diagnostics, self‑healing
  - `strategies/` — trading strategies and glue
  - `trading/` — exchange/Nautilus integrations and adapters
  - `utils/` — small utilities only (no domain logic)
- `tests/` — unit/integration/perf/stress suites
- `migrations/` — DB migrations (e.g., pgvector DDL variants)
- `Documentation/` — product, architecture, guides

## Current State vs Target

- Data Pipeline:
  - Present: `src/data_pipeline/enhanced_websocket_manager.py` and `src/data_pipeline/websocket_feed.py` overlap.
  - Target: consolidate into one manager with consistent config and subscriptions.

- Nautilus dependencies:
  - Present: `src/strategies/xgboost_nautilus_strategy.py` imports Nautilus at module import time.
  - Target: move heavy imports under guarded paths or add stubs; prefer adapters (`src/trading/nautilus_*`).

- Memory & Embeddings:
  - Present: `src/memory/memory_service.py`, `src/database/embedding_dao.py`, `EmbeddingRecord` model — aligned.
  - Target: migrate to pgvector ANN for similarity; keep API stable.

## Style & Standards

- No domain logic in `utils/`.
- Keep hot path (data→features→decision→execution) dependency‑minimal.
- Feature‑flag optional subsystems (graph/LLM/expensive)
- Keep API signatures stable; schedule breaking changes.

## Discrepancies to Address (Staged)

1) Duplicate WebSocket implementations
- Risk: drift and wasted maintenance.
- Action: Merge `websocket_feed.py` into `enhanced_websocket_manager.py` or vice versa. Create one config type and one subscription implementation.

2) Strategy hard dependency on Nautilus
- Risk: import failure in environments without Nautilus.
- Action: defer imports or stub Strategy base when missing; relocate Nautilus‑specific strategy into `strategies/nautilus/`.

3) Async persistence patterns
- Risk: blocking in event loops.
- Action: event‑loop safe scheduling added; extend with background task runner.

4) Metrics
- Risk: incomplete observability on hot path.
- Action: wire counters/histograms to `/api/v1/metrics` from WS→ML→execution.

## Refactor Plan (No file moves yet)

- Phase A (safe):
  - Align WebSocket config/types; add deprecation notices in duplicate module.
  - Add import guards for Nautilus‑dependent strategies.
- Phase B (moderate):
  - Consolidate WebSocket modules; update imports.
  - Move Nautilus‑specific strategies to `strategies/nautilus/`.
- Phase C (ANN):
  - Switch embedding queries to pgvector ANN.

All moves will be proposed via PRD tasks before code changes.

