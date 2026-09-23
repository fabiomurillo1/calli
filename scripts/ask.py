"""Ask a question and get back the most relevant ingested papers.

Requires that ingest.py and embed.py have already been run - this only
searches papers that are already in the database with embeddings.

Example:
    python scripts/ask.py "how do retrieval methods handle out-of-distribution queries?"
    python scripts/ask.py "transformer architectures for long documents" --top-k 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from calli.embeddings.specter2 import Specter2Embedder
from calli.retrieval.ask import ask
from calli.storage.db import get_connection, init_db

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Calli a research question")
    parser.add_argument("question", help="Your question, in natural language")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    conn = get_connection(db_path)
    init_db(conn)

    print("Loading SPECTER2...")
    embedder = Specter2Embedder()

    results = ask(conn, embedder, args.question, top_k=args.top_k)
    conn.close()

    if not results:
        print("No results - have you run ingest.py and embed.py yet?")
        return

    print(f"\nTop {len(results)} papers for: \"{args.question}\"\n")
    for i, paper in enumerate(results, start=1):
        print(f"{i}. [{paper['score']:.3f}] {paper['title']}")
        print(f"   {paper['arxiv_id']} - {paper['abs_url']}")
        print()


if __name__ == "__main__":
    main()