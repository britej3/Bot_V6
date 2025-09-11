BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Vector dimension = 10 (matches current feature expectations in tests)
CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    namespace VARCHAR(64) NOT NULL,
    item_id VARCHAR(128) NOT NULL,
    vector VECTOR(10) NOT NULL,
    metadata JSONB,
    CONSTRAINT _emb_ns_item_uc UNIQUE (namespace, item_id)
);

-- HNSW index for cosine searches; adjust parameters as needed
CREATE INDEX IF NOT EXISTS ix_embeddings_ns ON embeddings (namespace);
CREATE INDEX IF NOT EXISTS hnsw_embeddings_vector ON embeddings USING hnsw (vector vector_cosine_ops);

COMMIT;

