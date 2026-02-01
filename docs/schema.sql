CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_label TEXT NOT NULL,
    user_content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    cluster_depth INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE centroids (
    prompt_id INTEGER PRIMARY KEY,
    embedding_vector BLOB NOT NULL,
    dimensions INTEGER NOT NULL DEFAULT 384,
    FOREIGN KEY (prompt_id) REFERENCES prompts (id)
);
CREATE TABLE cluster_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id TEXT NOT NULL UNIQUE,
    cluster_label TEXT,
    level_0 TEXT,
    level_1 TEXT,
    level_2 TEXT,
    level_3 TEXT,
    level_4 TEXT,
    level_5 TEXT,
    level_6 TEXT,
    level_7 TEXT,
    level_8 TEXT,
    level_9 TEXT,
    depth INTEGER NOT NULL,
    prompt_count INTEGER DEFAULT 0
);
CREATE INDEX idx_prompts_cluster_id ON prompts(cluster_id);
CREATE INDEX idx_prompts_source ON prompts(source);
CREATE INDEX idx_prompts_cluster_label ON prompts(cluster_label);
CREATE INDEX idx_prompts_content_length ON prompts(content_length);
CREATE INDEX idx_cluster_hierarchy_cluster_id ON cluster_hierarchy(cluster_id);
CREATE INDEX idx_cluster_hierarchy_depth ON cluster_hierarchy(depth);