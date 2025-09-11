# Byterover Handbook

Generated: 2025-09-06

## Layer 1: System Overview

Purpose: Production-ready autonomous algorithmic crypto futures scalping bot with self-learning, risk management, and multi-exchange execution.

Tech Stack: Python 3.9+, Docker, MkDocs, Shell scripts; minor Go integration; pytest; pre-commit tooling; MkDocs for docs.

Architecture: Layered architecture with data pipeline, AI/ML engine, trading engine, risk management, monitoring, and supporting scripts/config.

Key Technical Decisions:
- Python-first implementation with modular src layout
- Docker and docker-compose for containerization
- MkDocs documentation site via `mkdocs.yml`
- Pre-commit, linting, and formatting configs present

Entry Points:
- Python scripts in `src/` and top-level demo utilities
- Dockerfiles for container builds
- MkDocs site via `mkdocs.yml`

---

## Layer 2: Module Map

Core Modules:
- Data Pipeline: real-time/validated ingestion and feature engineering
- AI/ML Engine: model training, inference, interpretation
- Trading Engine: strategies, execution, position/risk management

Data Layer:
- Config files in `config/`, data samples in `data/`

Integration Points:
- Exchange APIs via client libraries (Binance/OKX/Bybit)
- Optional external trading frameworks

Utilities:
- `scripts/`, `init-scripts/`, and monitoring helpers

Module Dependencies:
- Core modules depend on config, utils, and external libraries in `requirements*.txt`

---

## Layer 3: Integration Guide

API Endpoints:
- Not an HTTP service; integrates with exchange APIs

Configuration Files:
- `pyproject.toml`, `pytest.ini`, `requirements*.txt`, `mkdocs.yml`

External Integrations:
- Crypto exchange APIs; optional orchestration/monitoring components

Workflows:
- Local run via Python entry scripts, container builds via Dockerfiles

Interface Definitions:
- Module-level functions/classes with clear inputs/outputs in `src/`

---

## Layer 4: Extension Points

Design Patterns:
- Layered, modular design; configuration-driven components

Extension Points:
- Add new strategies under trading engine; extend data features; plug-in risk rules

Customization Areas:
- `config/` values, environment variables, and module composition

Plugin Architecture:
- Not formalized; modules designed for composability

Recent Changes:
- Repository preparation for GitHub and standardization

---

## Quality Validation Checklist (initial)
- [x] System Overview completed
- [x] Module Map completed
- [x] Integration Guide completed
- [x] Extension Points completed
- [x] Tech stack and architecture documented

Byterover handbook optimized for agent navigation and human developer onboarding

