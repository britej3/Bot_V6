# Junk Quarantine

This directory quarantines files that are unwanted for the Git repository or are slated for removal in the future. Items here are excluded by `.gitignore` and safe to delete later.

Moved on: $(date)

Policy:
- OS cruft (e.g., `.DS_Store`)
- Local secrets/env files (e.g., `.env`)
- Local databases and generated artifacts (e.g., `*.db`)
- Temporary files and logs

Contents:
- .DS_Store
- .env
- test.db
- .roo/mcp.json (contained API key)
