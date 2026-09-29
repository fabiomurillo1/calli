"""Sweep n_components_to_remove against a real question on YOUR actual
corpus, rather than guessing from synthetic simulations. Cheap to run -
reuses embeddings you already have, no re-ingestion or re-embedding.
"""
import argparse
from pathlib import Path

from calli.embeddings.specter2 import Specter2Embedder
from calli.storage.db import get_connection, get_papers_by_ids, init_db, search_similar

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep n_components_to_remove for a question")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH
    conn = get_connection(db_path)
    init_db(conn)

    print("Loading SPECTER2...")
    embedder = Specter2Embedder()
    query_vector = embedder.embed_query(args.question)

    for n in [0, 1, 2, 3, 5, 8, 15]:
        results = search_similar(conn, "specter2", query_vector, top_k=args.top_k, n_components_to_remove=n)
        metadata = get_papers_by_ids(conn, [pid for pid, _ in results])
        print(f"\n=== n_components_to_remove={n} ===")
        for pid, score in results:
            title = metadata.get(pid, {}).get("title", "?")
            print(f"  [{score:.3f}] {title[:70]}  ({pid})")

    conn.close()


if __name__ == "__main__":
    main()
