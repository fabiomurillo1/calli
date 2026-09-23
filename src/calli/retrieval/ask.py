"""Ties together embedding a question and ranking stored papers against it.

This is the first place in the codebase where a user's actual question
becomes ranked, readable results - everything before this (ingestion,
storage, embedding) was building the data this depends on.
"""
from __future__ import annotations

from calli.embeddings.specter2 import MODEL_NAME, Specter2Embedder
from calli.storage.db import get_papers_by_ids, search_similar
import sqlite3


def ask(
    conn: sqlite3.Connection,
    embedder: Specter2Embedder,
    question: str,
    top_k: int = 5,
    model: str = MODEL_NAME,
) -> list[dict]:
    """Given a natural-language question, return the top_k most relevant papers.

    Each result is a dict with the paper's metadata plus a "score" field
    (cosine similarity, roughly 0-1, higher is more relevant).
    """
    query_vector = embedder.embed_query(question)
    ranked = search_similar(conn, model, query_vector, top_k=top_k)

    metadata_by_id = get_papers_by_ids(conn, [paper_id for paper_id, _ in ranked])

    results = []
    for paper_id, score in ranked:
        paper = metadata_by_id.get(paper_id)
        if paper is None:
            continue  # embedding exists but paper metadata is missing somehow; skip rather than crash
        results.append({**paper, "score": score})
    return results