"""Diagnostic: check corpus composition and embedding coverage.

Distinguishes "the ranking logic is broken" from "the corpus doesn't have
enough on-topic papers to rank well" - two different problems that look
identical from ask.py's output alone.
"""
from pathlib import Path

from calli.storage.db import get_connection, init_db

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"

conn = get_connection(DB_PATH)
init_db(conn)

total_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
total_embedded = conn.execute(
    "SELECT COUNT(*) FROM paper_embeddings WHERE model = 'specter2'"
).fetchone()[0]
print(f"Total papers in DB: {total_papers}")
print(f"Total embedded (specter2): {total_embedded}")

print("\nPapers per primary_category:")
for row in conn.execute(
    "SELECT primary_category, COUNT(*) FROM papers GROUP BY primary_category ORDER BY COUNT(*) DESC"
):
    print(f"  {row[0]}: {row[1]}")

print("\nPapers with 'diffusion' in the title:")
rows = conn.execute(
    "SELECT arxiv_id, title FROM papers WHERE title LIKE '%diffusion%' COLLATE NOCASE"
).fetchall()
if rows:
    for r in rows:
        print(f"  [{r[0]}] {r[1]}")
else:
    print("  (none found)")

conn.close()