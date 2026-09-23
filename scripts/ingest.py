"""Ingest papers from arXiv into the local SQLite database.

Examples:
    python scripts/ingest.py "cat:cs.CL AND abs:retrieval" --max-results 50
    python scripts/ingest.py "au:Bengio_Y" --max-results 10 --db-path data/calli.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

from calli.ingestion.arxiv_client import ArxivClient
from calli.storage.db import get_connection, init_db, upsert_papers

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers from arXiv")
    parser.add_argument(
        "query",
        help="arXiv search query, e.g. 'cat:cs.CL AND abs:\"retrieval augmented generation\"'",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database (default: data/calli.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    client = ArxivClient()
    papers = client.search(args.query, max_results=args.max_results)

    conn = get_connection(db_path)
    init_db(conn)
    new_count, updated_count = upsert_papers(conn, papers)
    conn.close()

    print(f"Fetched {len(papers)} papers -> {db_path}")
    print(f"  new: {new_count}  updated: {updated_count}")


if __name__ == "__main__":
    main()