"""Ask a question and get back a synthesized, cited answer plus the papers behind it.

Requires that ingest.py and embed.py have already been run - this only
searches papers that are already in the database with embeddings. Also
requires ANTHROPIC_API_KEY to be set in the environment for the synthesis step.

Example:
    python scripts/ask.py "how do retrieval methods handle out-of-distribution queries?"
    python scripts/ask.py "transformer architectures for long documents" --top-k 3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from calli.embeddings.specter2 import Specter2Embedder
from calli.retrieval.ask import ask
from calli.storage.db import get_connection, init_db
from calli.synthesis.synthesize import synthesize_answer

load_dotenv()  # reads .env in the project root, if present; harmless if not

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Calli a research question")
    parser.add_argument("question", help="Your question, in natural language")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", default=None)
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip the LLM synthesis step and just show ranked papers",
    )
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

    if args.no_synthesis:
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("(Skipping synthesis: ANTHROPIC_API_KEY isn't set in the environment.)")
        return

    print("Synthesizing answer...\n")
    answer = synthesize_answer(args.question, results)
    print(answer)


if __name__ == "__main__":
    main()