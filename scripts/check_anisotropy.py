"""Diagnostic: check for embedding anisotropy.

If SPECTER2's raw embeddings cluster in a narrow cone (a known property of
BERT-family models), then even completely unrelated papers will show
artificially high cosine similarity to each other. This checks that
directly, rather than assuming it.
"""
from pathlib import Path

import numpy as np

from calli.storage.db import get_all_embeddings, get_connection, get_papers_by_ids, init_db

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "calli.db"

conn = get_connection(DB_PATH)
init_db(conn)

pairs = get_all_embeddings(conn, "specter2")
ids = [pid for pid, _ in pairs]
vectors = np.stack([vec for _, vec in pairs])
norm_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

metadata = get_papers_by_ids(conn, ids)
id_to_index = {pid: i for i, pid in enumerate(ids)}

# Baseline: pairwise similarity between random pairs, regardless of topic.
# If embeddings were well-separated by meaning, most of these should be
# low (unrelated topics). If they cluster high, that's anisotropy.
rng = np.random.default_rng(0)
n = len(ids)
sample_pairs = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(2000)]
sims = np.array([float(np.dot(norm_vectors[i], norm_vectors[j])) for i, j in sample_pairs])

print(f"Corpus size: {n} papers\n")
print("Random pairwise cosine similarity across the WHOLE corpus (any topic vs any topic):")
print(f"  mean: {sims.mean():.3f}   std: {sims.std():.3f}   min: {sims.min():.3f}   max: {sims.max():.3f}\n")

# Concrete example: an astro-ph paper vs a cs.CL paper should be close to
# unrelated if the embedding space is healthy.
astro_id = next((pid for pid in ids if metadata[pid]["primary_category"].startswith("astro-ph")), None)
cl_id = next((pid for pid in ids if metadata[pid]["primary_category"] == "cs.CL"), None)

if astro_id and cl_id:
    sim = float(np.dot(norm_vectors[id_to_index[astro_id]], norm_vectors[id_to_index[cl_id]]))
    print("Similarity between an astro-ph paper and a cs.CL paper (should be LOW if embeddings are healthy):")
    print(f"  [{astro_id}] {metadata[astro_id]['title']}")
    print(f"  [{cl_id}] {metadata[cl_id]['title']}")
    print(f"  cosine similarity: {sim:.3f}")

conn.close()