-- Calli: metadata storage schema (SQLite)
-- This covers paper metadata only. Embeddings/vector storage is a separate
-- decision (comes with the RAG architecture discussion) and deliberately
-- isn't modeled here yet.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Core paper metadata, one row per arXiv paper (keyed by arxiv_id, which
-- includes the version suffix e.g. "2005.11401v4" so re-ingesting an
-- updated version doesn't silently collide with the old one).
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    primary_category  TEXT NOT NULL,
    published         TEXT NOT NULL,   -- ISO 8601 string
    updated           TEXT NOT NULL,   -- ISO 8601 string
    pdf_url           TEXT NOT NULL,
    abs_url           TEXT NOT NULL,
    comment           TEXT,
    journal_ref       TEXT,
    doi               TEXT,
    ingested_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Authors, deduplicated by name so "who else has this author written with"
-- style queries are possible later.
CREATE TABLE IF NOT EXISTS authors (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE
);

-- Junction table preserving author order as listed on the paper.
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id      TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    author_id     INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    author_order  INTEGER NOT NULL,
    PRIMARY KEY (paper_id, author_id)
);

-- arXiv category codes (e.g. "cs.CL", "cs.LG"). Small, mostly-fixed
-- vocabulary, so a lookup table is cheap and makes filtering trivial.
CREATE TABLE IF NOT EXISTS categories (
    code  TEXT PRIMARY KEY
);

-- Junction: a paper can belong to multiple categories, one of which is
-- also recorded as primary_category on the papers row for quick access.
CREATE TABLE IF NOT EXISTS paper_categories (
    paper_id       TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    category_code  TEXT NOT NULL REFERENCES categories(code) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, category_code)
);

-- Embeddings for papers, keyed by (paper, model) so a paper can hold
-- multiple embeddings if you ever compare models or re-embed later.
-- Kept as its own table (not a column on papers) for exactly that reason.
CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id    TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (paper_id, model)
);

CREATE INDEX IF NOT EXISTS idx_paper_embeddings_model ON paper_embeddings(model);

-- Indexes for the query patterns we already know we'll need.
CREATE INDEX IF NOT EXISTS idx_papers_primary_category ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_paper_categories_category ON paper_categories(category_code);