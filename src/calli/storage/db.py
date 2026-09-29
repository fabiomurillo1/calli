"""SQLite storage layer for paper metadata.

Keeps all DB access in one place so ingestion, search, and anything else
that needs papers share the same connection/upsert logic rather than each
writing their own SQL.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from calli.models.paper import Paper

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path("data") / "calli.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with sane defaults (FK enforcement, WAL mode)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables/indexes if they don't already exist. Safe to call every run."""
    conn.executescript(SCHEMA_PATH.read_text())


def paper_exists(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return row is not None


def upsert_paper(conn: sqlite3.Connection, paper: Paper) -> bool:
    """Insert a paper, or update it in place if it already exists.

    Returns True if this was a new paper, False if it already existed.
    """
    is_new = not paper_exists(conn, paper.arxiv_id)

    with conn:
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, title, abstract, primary_category,
                published, updated, pdf_url, abs_url,
                comment, journal_ref, doi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                primary_category=excluded.primary_category,
                published=excluded.published,
                updated=excluded.updated,
                pdf_url=excluded.pdf_url,
                abs_url=excluded.abs_url,
                comment=excluded.comment,
                journal_ref=excluded.journal_ref,
                doi=excluded.doi
            """,
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                paper.primary_category,
                paper.published.isoformat(),
                paper.updated.isoformat(),
                paper.pdf_url,
                paper.abs_url,
                paper.comment,
                paper.journal_ref,
                paper.doi,
            ),
        )

        for category_code in paper.categories:
            conn.execute("INSERT OR IGNORE INTO categories (code) VALUES (?)", (category_code,))
            conn.execute(
                "INSERT OR IGNORE INTO paper_categories (paper_id, category_code) VALUES (?, ?)",
                (paper.arxiv_id, category_code),
            )

        for order, author_name in enumerate(paper.authors):
            conn.execute("INSERT OR IGNORE INTO authors (name) VALUES (?)", (author_name,))
            author_id = conn.execute(
                "SELECT id FROM authors WHERE name = ?", (author_name,)
            ).fetchone()[0]
            conn.execute(
                """
                INSERT OR IGNORE INTO paper_authors (paper_id, author_id, author_order)
                VALUES (?, ?, ?)
                """,
                (paper.arxiv_id, author_id, order),
            )

    return is_new


def upsert_papers(conn: sqlite3.Connection, papers: Iterable[Paper]) -> tuple[int, int]:
    """Upsert many papers at once. Returns (new_count, updated_count)."""
    new_count = 0
    updated_count = 0
    for paper in papers:
        if upsert_paper(conn, paper):
            new_count += 1
        else:
            updated_count += 1
    return new_count, updated_count


# --- Embeddings -------------------------------------------------------
# Vectors are stored as raw float32 bytes rather than JSON/text: it's
# smaller, avoids floating-point round-tripping issues, and numpy can
# read it straight back with no parsing step.


def upsert_embedding(conn: sqlite3.Connection, paper_id: str, model: str, vector: np.ndarray) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    with conn:
        conn.execute(
            """
            INSERT INTO paper_embeddings (paper_id, model, embedding, dim)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id, model) DO UPDATE SET
                embedding=excluded.embedding,
                dim=excluded.dim,
                created_at=datetime('now')
            """,
            (paper_id, model, vector.tobytes(), vector.shape[0]),
        )


def get_embedding(conn: sqlite3.Connection, paper_id: str, model: str) -> np.ndarray | None:
    row = conn.execute(
        "SELECT embedding, dim FROM paper_embeddings WHERE paper_id = ? AND model = ?",
        (paper_id, model),
    ).fetchone()
    if row is None:
        return None
    blob, dim = row
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def get_all_embeddings(conn: sqlite3.Connection, model: str) -> list[tuple[str, np.ndarray]]:
    """Return (paper_id, vector) for every paper embedded with `model`."""
    rows = conn.execute(
        "SELECT paper_id, embedding, dim FROM paper_embeddings WHERE model = ?",
        (model,),
    ).fetchall()
    return [(pid, np.frombuffer(blob, dtype=np.float32, count=dim)) for pid, blob, dim in rows]


def papers_missing_embedding(conn: sqlite3.Connection, model: str) -> list[tuple[str, str, str]]:
    """(arxiv_id, title, abstract) for papers that don't yet have an embedding for `model`."""
    return conn.execute(
        """
        SELECT p.arxiv_id, p.title, p.abstract
        FROM papers p
        LEFT JOIN paper_embeddings pe
            ON pe.paper_id = p.arxiv_id AND pe.model = ?
        WHERE pe.paper_id IS NULL
        """,
        (model,),
    ).fetchall()


def get_papers_by_ids(conn: sqlite3.Connection, arxiv_ids: list[str]) -> dict[str, dict]:
    """Fetch full metadata for a set of papers, keyed by arxiv_id.

    Used to turn search_similar's (paper_id, score) pairs into something
    actually readable - title, abstract, links - rather than bare IDs.
    """
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" for _ in arxiv_ids)
    rows = conn.execute(
        f"""
        SELECT arxiv_id, title, abstract, primary_category, pdf_url, abs_url
        FROM papers
        WHERE arxiv_id IN ({placeholders})
        """,
        arxiv_ids,
    ).fetchall()
    return {
        row[0]: {
            "arxiv_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "primary_category": row[3],
            "pdf_url": row[4],
            "abs_url": row[5],
        }
        for row in rows
    }


def search_similar(
    conn: sqlite3.Connection,
    model: str,
    query_vector: np.ndarray,
    top_k: int = 5,
    n_components_to_remove: int = 3,
) -> list[tuple[str, float]]:
    """Cosine similarity search over stored embeddings, as one matrix operation.

    Loads every vector for `model` into memory and computes all similarities
    at once via numpy rather than looping in Python. At 768 dimensions,
    float32, even 100k papers is ~300MB in memory and this runs in well
    under a second - fine for a corpus far bigger than you'll have for a
    while. If the corpus grows into the hundreds of thousands and this
    becomes measurably slow, that's the point to look at sqlite-vec (an
    ANN-search SQLite extension - stays in-process, no new service) before
    reaching for something heavier like FAISS or a separate vector DB.

    Two corrections are applied before comparing, both addressing the same
    root problem: raw transformer embeddings like SPECTER2's share a large,
    content-unrelated "background" that dominates cosine similarity and
    buries the actual topical signal (a well-documented property of this
    embedding family, not specific to this corpus).

    1. Mean-centering: subtract the corpus's mean embedding from every
       vector (including the query). Removes a constant shared bias.
    2. Top-component removal: on top of that, strip out the
       `n_components_to_remove` directions of highest variance across the
       corpus. Anisotropy isn't always just a constant offset - it often
       shows up as a handful of shared directions the whole corpus varies
       along regardless of topic. These are fit on the corpus alone (never
       on the query), so the query doesn't influence what counts as
       "background" - it just gets the same correction applied afterward.

    Both are recomputed from whatever's currently in the corpus each call,
    so they stay accurate as more papers get embedded over time.
    """
    pairs = get_all_embeddings(conn, model)
    if not pairs:
        return []

    paper_ids = [paper_id for paper_id, _ in pairs]
    matrix = np.stack([vector for _, vector in pairs])  # shape: (n_papers, dim)

    query_vector = np.asarray(query_vector, dtype=np.float32)

    corpus_mean = matrix.mean(axis=0)
    matrix_centered = matrix - corpus_mean
    query_centered = query_vector - corpus_mean

    # Fit top variance directions on the corpus only, then remove them from
    # both the corpus and the query. Capped relative to corpus size (not
    # just to n_components_to_remove) because removing even a handful of
    # components from a small corpus can consume most of its usable rank
    # and strip out real signal along with the noise - e.g. 4 papers span
    # at most a 3-dimensional space after centering, so "remove 3" would
    # wipe out everything. Below roughly 10 papers per component, skip
    # this step and rely on mean-centering alone.
    max_components = min(
        n_components_to_remove,
        matrix_centered.shape[0] // 10,
        matrix_centered.shape[1],
    )
    if max_components > 0:
        _, _, top_directions = np.linalg.svd(matrix_centered, full_matrices=False)
        for direction in top_directions[:max_components]:
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            matrix_centered = matrix_centered - np.outer(matrix_centered @ direction, direction)
            query_centered = query_centered - (query_centered @ direction) * direction

    query_norm = query_centered / (np.linalg.norm(query_centered) + 1e-8)
    matrix_norms = matrix_centered / (np.linalg.norm(matrix_centered, axis=1, keepdims=True) + 1e-8)

    similarities = matrix_norms @ query_norm  # shape: (n_papers,), one matmul

    top_indices = np.argsort(-similarities)[:top_k]
    return [(paper_ids[i], float(similarities[i])) for i in top_indices]


_STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "how", "what", "why", "of",
    "in", "on", "for", "to", "and", "or", "with", "this", "that", "these",
    "those", "work", "works", "about", "can", "you",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def search_hybrid(
    conn: sqlite3.Connection,
    model: str,
    query_vector: np.ndarray,
    query_text: str,
    top_k: int = 5,
    n_components_to_remove: int = 3,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Combine embedding similarity with literal keyword overlap.

    Dense embeddings are good at conceptual similarity but can be fooled by
    incidental structural resemblance - e.g. a paper with zero topical
    connection to the query outranking one that literally has the query's
    key terms in its title. This blends an embedding-based ranking with a
    simple lexical-overlap ranking using Reciprocal Rank Fusion (RRF):
    each paper gets a rank from both methods, and the two ranks are
    combined rather than the raw scores. RRF is used specifically because
    it sidesteps having to normalize two very different, incomparable
    score scales (cosine similarity vs. a keyword-overlap count) - only
    the relative ORDER from each method matters, not the magnitude.
    """
    pairs = get_all_embeddings(conn, model)
    if not pairs:
        return []

    all_ids = [pid for pid, _ in pairs]

    # Full embedding ranking (same corrected similarity as search_similar,
    # just requesting every paper instead of only the top_k).
    embedding_ranked = search_similar(
        conn, model, query_vector, top_k=len(all_ids), n_components_to_remove=n_components_to_remove
    )
    embedding_rank = {paper_id: rank for rank, (paper_id, _) in enumerate(embedding_ranked)}

    # Full lexical ranking: how many distinct meaningful query terms appear
    # in each paper's title + abstract.
    query_terms = _tokenize(query_text)
    metadata = get_papers_by_ids(conn, all_ids)
    lexical_scores = []
    for paper_id in all_ids:
        paper = metadata.get(paper_id)
        text = f"{paper['title']} {paper['abstract']}" if paper else ""
        overlap = len(query_terms & _tokenize(text))
        lexical_scores.append((paper_id, overlap))
    lexical_scores.sort(key=lambda item: -item[1])
    lexical_rank = {paper_id: rank for rank, (paper_id, _) in enumerate(lexical_scores)}

    combined = []
    for paper_id in all_ids:
        rrf_score = (
            1.0 / (rrf_k + embedding_rank.get(paper_id, len(all_ids)))
            + 1.0 / (rrf_k + lexical_rank.get(paper_id, len(all_ids)))
        )
        combined.append((paper_id, rrf_score))

    combined.sort(key=lambda item: -item[1])
    return combined[:top_k]