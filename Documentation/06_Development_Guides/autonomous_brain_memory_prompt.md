# Autonomous Self-Learning Crypto Futures Trading Bot Workflow (Brain-like Memory System)

Goal: Build a fully autonomous, AI-enhanced trading system with layered memory, continuous learning, robust error recovery, and end-to-end operations. Prioritize completing non-ML plumbing (data, validation, observability, safety) before enabling ML retraining or live-risk changes.

## Objectives
- Self-learning: Retain state/action/outcome for iterative improvement and regime-aware decisions.
- Self-adapting: Adjust strategy parameters and model weights when performance degrades.
- Self-healing: Detect faults, roll back to last-known-good configurations, and recover safely.
- Research/backtesting/hyperoptimization: Continuously ingest research, evaluate via walk-forward, and tune parameters.

## Memory Architecture
- Episodic: Trades, decisions, outcomes with timestamps and market context.
- Semantic: Domain knowledge, exchange specs, strategy rules, risk limits, playbooks.
- Procedural: Runbooks for failover, rollback, retraining, deployment.
- Vector (Qdrant/pgvector): Embeddings for state-action pairs, regimes, hyperopt results.
- Graph (Neo4j/LightRAG/Graphiti): Entities and relations for market events, signals, parameter interactions.

## Orchestration (n8n)
1. Ingest & Normalize: Multi-exchange WS; dedup/order; validate.
2. Persist: Embed to vector DB; update graph entities/edges.
3. Research: Periodic ingestion of papers/strategies; embed and tag.
4. Backtest: Walk-forward (purged); per-regime evaluation; aggregate reports.
5. Hyperopt: Constrained search; time-boxed; persist top-k params & metadata.
6. Prediction: Select model/params by regime & similarity; infer under latency SLA.
7. Execution: Route orders; enforce risk & circuit breakers; log actions/results.
8. Evaluation: Monitor PnL/latency/slippage; detect degradation.
9. Adaptation: Retrieve best historical analogs; propose changes; revalidate; promote if safe.
10. Self-healing: Roll back to stable versions or safe strategy; log incident; notify.
11. Reporting: Daily/weekly performance and governance reports.

## Readiness Gates
- Data Quality: Validator pass rate > target.
- Observability: WS reconnects within threshold; no stale feeds; metrics healthy.
- Backtests: Walk-forward outperforming baseline with statistical significance.
- Governance: Changes logged with versioned artifacts and rollback plan.

## Policies
- Never deploy high-risk changes without passing gates and backtest validation.
- Record versioned artifacts (model files, params, reports).
- Prefer safe fallback over uncertain adaptation; exploratory trades are limited-risk.

## Repo Integration
- Data/WS: `src/data_pipeline/enhanced_websocket_manager.py`, `src/data_pipeline/websocket_feed.py`
- Validation: `src/data_pipeline/data_validator.py`
- Backtesting: `src/validation/walk_forward_backtester.py`
- HPO Stub: `src/validation/hpo_study_stub.py`
- Ensemble Persistence: `src/learning/xgboost_ensemble.py`
- API: `/api/v1/backtest/walk-forward/synthetic`, `/api/v1/metrics`

Use this document as the operational prompt/checklist for orchestration and governance.

