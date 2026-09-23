"""Embed papers already in the database using SPECTER2.

Run this after ingest.py has populated the `papers` table. It only embeds
papers that don't already have an embedding for this model, so it's safe
to re-run any time after new papers get ingested.

Example:
    python scripts/embed.py
    python scripts/embed.py --batch-size 4 --db-path data/calli.db
"""
from __future__ import annotations
import argparse
from pathlib import Path

from calli.embeddings.specter2 import MODEL_NAME, Specter2Embedder
from calli.storage.db import get_connection, init_db, papers_missing_embedding, upsert_embedding

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed papers with SPECTER2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--db-path", default=None, help="Path to the SQLite database")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    conn = get_connection(db_path)
    init_db(conn)

    rows = papers_missing_embedding(conn, MODEL_NAME)
    if not rows:
        print("Nothing to embed - every paper already has an embedding.")
        return

    print(f"Loading {MODEL_NAME} locally (first run downloads the model, ~440MB)...")
    embedder = Specter2Embedder()

    print(f"Embedding {len(rows)} paper(s)...")
    texts = [embedder.document_text(title, abstract) for _, title, abstract in rows]
    vectors = embedder.embed_documents(texts, batch_size=args.batch_size)

    for (arxiv_id, _, _), vector in zip(rows, vectors):
        upsert_embedding(conn, arxiv_id, MODEL_NAME, vector)

    conn.close()
    print(f"Embedded and stored {len(rows)} paper(s) -> {db_path}")


if __name__ == "__main__":
    main()