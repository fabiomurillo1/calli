"""Domain model for a paper retrieved from arXiv."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Paper(BaseModel):
    """A single paper as returned by the arXiv API.

    Kept intentionally close to what arXiv actually gives us; higher-level
    concerns (embeddings, tags the user adds, relevance scores, etc.) should
    live in separate models that reference `arxiv_id`, not bolted on here.
    """

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    def __str__(self) -> str:  # nicer for logging/debugging
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += " et al."
        return f"[{self.arxiv_id}] {self.title} ({authors})"