# Memory Policy

This workspace uses the Knowledge Graph MCP Server as the default memory layer for storing research, implementation decisions, validation results, and operational playbooks.

## Default Memory
- Primary: Knowledge Graph (MCP `server-memory`).
- Scope: Non-secret technical knowledge, plans, pitfalls, and runbooks.
- Exclusions: Do not store secrets, tokens, private keys, or PII.

## Usage Guidelines
- Store reusable insights with precise context and minimal ambiguity.
- Prefer compact, high-signal observations; include exact code/paths when helpful.
- Keep label cardinality low for metrics and taxonomy consistency.

## Migration From ByteRover Memory
- Status: Pending. Access to Byterover memory requires authentication in the ByteRover extension. Once authenticated:
  1. Export or retrieve relevant knowledge using Byterover memory tools.
  2. Normalize content (remove secrets) and store into the Knowledge Graph.
  3. Mark entities with a `source: byterover` attribute.

## Operational Notes
- Retrieval priority: Knowledge Graph → Repo files → Web.
- When conflicts arise, record resolution rationale and link to authoritative source.
