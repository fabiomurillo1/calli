"""SPECTER2 embedding wrapper.

SPECTER2 (Allen Institute for AI) is a SciBERT-based model trained on
scientific papers using citation relationships as a training signal, which
is why it's a better fit here than a general-purpose text embedding model:
its notion of "similar" was learned from how papers actually cite each
other, not just shared vocabulary.

It ships as one base model plus swappable adapters for different tasks.
We load two of them and switch between them depending on what's being
embedded:
    - "allenai/specter2"             -> encoding papers (title + abstract)
    - "allenai/specter2_adhoc_query"  -> encoding a natural-language question

Using a different adapter for queries than for documents is the setup
Allen AI documents for ad hoc search (question -> retrieve documents), as
opposed to paper-to-paper similarity, which would use the document
adapter on both sides.

Model weights download from Hugging Face on first use and are cached
locally afterward (~440MB for the base model, adapters are a few MB each).
"""
from __future__ import annotations

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import logging

logging.getLogger("adapters").setLevel(logging.ERROR)

BASE_MODEL = "allenai/specter2_base"
DOCUMENT_ADAPTER = "allenai/specter2"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"
MAX_LENGTH = 512

# What we store in the `model` column in paper_embeddings. Bump this if
# you ever change adapters/base model, so old and new vectors don't get
# compared as if they lived in the same space.
MODEL_NAME = "specter2"


class Specter2Embedder:
    """Loads SPECTER2 once; swaps the active adapter per call."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        self.model.load_adapter(DOCUMENT_ADAPTER, source="hf", load_as="document", set_active=False)
        self.model.load_adapter(QUERY_ADAPTER, source="hf", load_as="query", set_active=False)
        self.model.eval()

    def embed_documents(self, texts: list[str], batch_size: int = 8) -> list[np.ndarray]:
        """Embed papers. `texts` should be "title <SEP> abstract" per SPECTER2 convention."""
        self.model.set_active_adapters("document")
        return self._embed_texts(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a natural-language question into the same space as embed_documents."""
        self.model.set_active_adapters("query")
        return self._embed_texts([text], batch_size=1)[0]

    def _embed_texts(self, texts: list[str], batch_size: int) -> list[np.ndarray]:
        all_vectors: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                output = self.model(**inputs)
                # SPECTER2 uses the [CLS] token (first position) as the
                # document/query representation.
                cls_vectors = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_vectors.extend(v.astype(np.float32) for v in cls_vectors)
        return all_vectors

    def document_text(self, title: str, abstract: str) -> str:
        """Build the "title <SEP> abstract" text SPECTER2 expects for a paper."""
        return f"{title}{self.tokenizer.sep_token}{abstract}"