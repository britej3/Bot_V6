# CryptoScalp AI Project Structure

## Overview

This document outlines the current project structure for the CryptoScalp AI autonomous algorithmic high-frequency crypto futures scalping bot. The project follows a modular architecture with clear separation of concerns.

## Root Level Structure

```
📦 Bot_V5/
├── 📄 ARCHIVED_CONCEPTS.md          # Deprecated components and historical decisions
├── 📄 CHANGELOG.md                   # Version history and updates
├── 📄 ENHANCED_ARCHITECTURE_V2.md   # Current state-of-the-art architecture
├── 📄 IMPLEMENTATION_PLAN.md         # Implementation roadmap and milestones
├── 📄 MINIMUM_VIABLE_SLICE.md        # MVVS implementation guide
├── 📄 PRD_TASK_BREAKDOWN.md          # Task breakdown (to be migrated to Jira)
├── 📄 PROJECT_MIGRATION_PLAN.md      # Jira/Confluence migration plan
├── 📄 Plan.md                        # Original project plan (historical)
├── 📄 PRD.md                         # Product requirements document
├── 📄 index.md                       # MkDocs main page
├── 📄 mkdocs.yml                     # Documentation configuration
├── 📄 pyproject.toml                 # Python project configuration
├── 📄 pytest.ini                     # Test configuration
├── 📄 requirements*.txt              # Python dependencies
├── 📁 archive/                       # Historical documentation
├── 📁 data/                          # Data storage and processing
├── 📁 docs/                          # MkDocs documentation files
├── 📁 Documentation/                 # Comprehensive documentation structure
├── 📁 init-scripts/                  # Database initialization scripts
├── 📁 logs/                          # Application logs
├── 📁 models/                        # Trained model storage
├── 📁 monitoring/                    # Monitoring and observability configs
├── 📁 src/                           # Source code
├── 📁 tests/                         # Test suites
├── 📁 miscellaneous/                 # Temporary and cache files
├── 📁 .github/                       # GitHub configurations and workflows
├── 📁 .vscode/                       # VSCode workspace settings
└── 📁 .roo/                          # Roo mode configurations
```

## Source Code Structure (`src/`)

```
📁 src/
├── 📄 __init__.py
├── 📄 config.py                      # Application configuration
├── 📄 main.py                        # Application entry point
├── 📁 api/                           # API endpoints and services
├── 📁 data_pipeline/                 # Data processing pipeline (deprecated)
├── 📁 models/                        # AI/ML models and components
│   ├── 📄 mixture_of_experts.py      # MoE architecture implementation
│   ├── 📄 mlops_manager.py          # MLOps functionality
│   ├── 📄 model_optimizer.py        # Model optimization utilities
│   └── 📄 self_awareness.py         # Self-awareness features
├── 📁 trading/                       # Trading logic and execution
└── 📁 utils/                         # Utility functions and helpers
```

## Documentation Structure (`Documentation/`)

```
📁 Documentation/
├── 📁 01_Project_Overview/           # Project vision and overview
├── 📁 02_Requirements/               # Functional and non-functional requirements
├── 📁 03_Architecture_Design/        # System architecture documentation
├── 📁 04_Database_Schema/            # Database design and schema
├── 📁 05_API_Documentation/          # API specifications and documentation
├── 📁 06_Development_Guides/         # Development environment and processes
├── 📁 07_Testing/                    # Testing strategies and procedures
├── 📁 08_Deployment/                 # Deployment guides and configurations
├── 📁 09_Maintenance/                # System maintenance procedures
├── 📁 10_Standards_and_Best_Practices/ # Coding standards and best practices
├── 📁 explanation/                   # Architecture decision records
├── 📁 how_to_guides/                 # Step-by-step guides
├── 📁 reference/                     # Technical reference materials
└── 📁 tutorials/                     # Learning tutorials
```

## Test Structure (`tests/`)

```
📁 tests/
├── 📄 __init__.py
├── 📄 conftest.py                    # Test configuration and fixtures
├── 📁 e2e/                           # End-to-end tests
├── 📁 fixtures/                      # Test data and fixtures
├── 📁 integration/                   # Integration tests
├── 📁 unit/                          # Unit tests
└── 📁 utils/                         # Test utilities
```

## Key Files Description

### Core Documentation
- **`ENHANCED_ARCHITECTURE_V2.md`**: Current architecture with MoE, optimization, and self-awareness features
- **`MINIMUM_VIABLE_SLICE.md`**: Implementation guide for validating one end-to-end trading path
- **`PROJECT_MIGRATION_PLAN.md`**: Plan for migrating from static docs to dynamic project management tools

### Implementation Files
- **`PRD_TASK_BREAKDOWN.md`**: Detailed task breakdown (bootstrap document, to be migrated to Jira)
- **`IMPLEMENTATION_PLAN.md`**: Implementation phases and timelines
- **`ARCHIVED_CONCEPTS.md`**: Documentation of deprecated components and migration paths

### Configuration Files
- **`mkdocs.yml`**: Documentation site configuration
- **`pyproject.toml`**: Python project metadata and dependencies
- **`docker-compose.yml`**: Docker services configuration
- **`.github/workflows/deploy-docs.yml`**: CI/CD pipeline for documentation

## File Organization Principles

### 1. Single Source of Truth
- Each concept has one authoritative documentation file
- Redundant files are moved to `archive/` with clear deprecation notices
- Cross-references maintained between related documents

### 2. Logical Grouping
- Related files grouped in appropriate directories
- Clear naming conventions for easy discovery
- Separation of concerns between documentation, implementation, and configuration

### 3. Version Control
- Historical versions preserved in `archive/` folder
- Clear migration paths documented
- Deprecation notices for superseded components

### 4. Accessibility
- Documentation organized by user type (tutorials, how-to, reference, explanation)
- Clear navigation paths through documentation structure
- Searchable content with consistent terminology

## Recent Changes

### Files Moved to Archive
- `ENHANCED_ARCHITECTURE.md` → `archive/ENHANCED_ARCHITECTURE.md` (superseded by V2)
- `ENHANCED_TECH_STACK.md` → `archive/ENHANCED_TECH_STACK.md` (consolidated into architecture docs)

### Files Moved to Miscellaneous
- `nohup.out` → `miscellaneous/nohup.out` (temporary output file)
- `.pytest_cache/` → `miscellaneous/.pytest_cache/` (test cache)
- `.kilocode/` → `miscellaneous/.kilocode/` (IDE cache)
- `.cursor/` → `miscellaneous/.cursor/` (IDE cache)

### New Files Added
- `MINIMUM_VIABLE_SLICE.md`: Implementation guide for MVVS approach
- `PROJECT_MIGRATION_PLAN.md`: Jira/Confluence migration strategy
- `ARCHIVED_CONCEPTS.md`: Historical record of deprecated components
- `.github/workflows/deploy-docs.yml`: Automated documentation deployment

## Maintenance Guidelines

### Adding New Files
1. Determine appropriate location based on file type and purpose
2. Follow existing naming conventions
3. Update this document to reflect changes
4. Add cross-references to related files if needed

### Moving Files
1. Update all references to moved files
2. Update import statements in code if applicable
3. Update documentation links
4. Test that moved files are accessible from new locations

### Deprecating Files
1. Add deprecation notice to file header
2. Move to `archive/` folder
3. Update `ARCHIVED_CONCEPTS.md` with migration guidance
4. Update references to point to replacement files

## Contact

For questions about project structure or file organization, contact the development team lead.