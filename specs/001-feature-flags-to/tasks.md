# Tasks: Feature flags to gate graph integration

**Input**: Design documents from `/specs/001-feature-flags-to/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Web app**: `backend/src/`, `frontend/src/`
- Paths shown below assume web app structure based on plan.md

## Phase 3.1: Setup
- [x] T001 Create `backend/src/config/feature_flags.json` with initial content `{"GRAPH_INTEGRATION_ENABLED": {"enabled": true, "description": "Enable/disable graph database integration."}}`

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T002 [P] Contract test for `GET /feature-flags` in `backend/tests/contract/test_feature_flags_get.py`
- [x] T003 [P] Contract test for `PUT /feature-flags/{name}` in `backend/tests/contract/test_feature_flags_put.py`
- [x] T004 [P] Integration test for feature flag service in `backend/tests/integration/test_feature_flag_service.py` to verify loading and reloading of flags from the config file.

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T005 [P] Create `FeatureFlag` Pydantic model in `backend/src/models/feature_flag.py` based on `data-model.md`.
- [x] T006 Create `FeatureFlagService` in `backend/src/services/feature_flag_service.py`. This service will be responsible for loading, reloading, and providing access to feature flags from `config/feature_flags.json`.
- [x] T007 Implement `GET /feature-flags` endpoint in `backend/src/api/feature_flags.py`.
- [x] T008 Implement `PUT /feature-flags/{name}` endpoint in `backend/src/api/feature_flags.py`.

## Phase 3.4: Integration
- [x] T009 Modify `MemoryService` in `backend/src/services/memory_service.py` to use the `FeatureFlagService` to check the `GRAPH_INTEGRATION_ENABLED` flag before routing requests to the graph database.

## Phase 3.5: Polish
- [x] T010 [P] Add unit tests for the `FeatureFlagService` in `backend/tests/unit/test_feature_flag_service.py`.
- [x] T011 [P] Update API documentation in `docs/api.md` to include the new feature flag endpoints.

## Dependencies
- T001 before T004, T006
- T002, T003, T004 before T005, T006, T007, T008
- T005 before T006
- T006 before T007, T008, T009
- T007, T008 before T011
- T009 before T010

## Parallel Example
```
# Launch T002-T004 together:
Task: "Contract test for GET /feature-flags in backend/tests/contract/test_feature_flags_get.py"
Task: "Contract test for PUT /feature-flags/{name} in backend/tests/contract/test_feature_flags_put.py"
Task: "Integration test for feature flag service in backend/tests/integration/test_feature_flag_service.py"
```