-- Enable pgvector and create embeddings table using native vector type
-- Adjust dimension to match your feature dimension (e.g., 128/256). Use the same namespace/item_id uniqueness.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Example with 256-dim vectors; change if needed
CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace VARCHAR(64) NOT NULL,
    item_id VARCHAR(128) NOT NULL,
    vector VECTOR(256) NOT NULL,
    metadata JSONB,
    CONSTRAINT _emb_ns_item_uc UNIQUE (namespace, item_id)
);

-- HNSW or IVFFlat index (choose based on workload). IVFFlat requires ANALYZE and lists tuning.
-- Example IVFFlat index; adjust lists according to data size (sqrt(n) is a rough starting point).
CREATE INDEX IF NOT EXISTS ix_embeddings_ns ON embeddings (namespace);
CREATE INDEX IF NOT EXISTS ivfflat_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

COMMIT;

