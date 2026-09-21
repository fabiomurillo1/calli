"""Thin, well-behaved client for arXiv's public Atom API.

Docs: https://info.arxiv.org/help/api/user-manual.html

Notes on the query syntax (search_query param), since it trips people up:
    - Field prefixes: ti: (title), abs: (abstract), au: (author), cat: (category)
    - Boolean ops: AND, OR, ANDNOT (must be uppercase)
    - Example: 'cat:cs.CL AND abs:"retrieval augmented generation"'
"""
from __future__ import annotations

import time
from typing import Literal

import feedparser
import requests

from calli.models.paper import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"

SortBy = Literal["relevance", "lastUpdatedDate", "submittedDate"]
SortOrder = Literal["ascending", "descending"]


class ArxivClient:
    """Queries the arXiv API and returns parsed `Paper` objects.

    arXiv asks API consumers to keep requests to roughly one every 3
    seconds. This client enforces that automatically across calls made on
    the same instance, so callers don't have to think about it.
    """

    def __init__(self, delay_seconds: float = 3.0, timeout: int = 30) -> None:
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self._last_request_time: float | None = None

    def search(
        self,
        query: str,
        max_results: int = 20,
        start: int = 0,
        sort_by: SortBy = "relevance",
        sort_order: SortOrder = "descending",
    ) -> list[Paper]:
        """Run a search query against arXiv and return matching papers."""
        self._respect_rate_limit()

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        return [self._entry_to_paper(entry) for entry in feed.entries]

    def get_by_id(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by its arXiv id, e.g. '2301.12345'."""
        results = self.search(query=f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    def iter_all(self, query: str, page_size: int = 100, max_total: int | None = None):
        """Generator that pages through all results for a query.

        Useful for bulk ingestion (e.g. "everything in cs.CL from the last
        month") without holding it all in memory or writing paging logic
        at every call site.
        """
        fetched = 0
        start = 0
        while True:
            remaining = None if max_total is None else max_total - fetched
            batch_size = page_size if remaining is None else min(page_size, remaining)
            if batch_size <= 0:
                return

            batch = self.search(query, max_results=batch_size, start=start)
            if not batch:
                return

            for paper in batch:
                yield paper

            fetched += len(batch)
            start += len(batch)
            if max_total is not None and fetched >= max_total:
                return

    def _respect_rate_limit(self) -> None:
        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _entry_to_paper(entry) -> Paper:
        arxiv_id = entry.id.split("/abs/")[-1]

        pdf_url = next(
            (link.href for link in entry.links if link.get("title") == "pdf"),
            entry.id.replace("/abs/", "/pdf/"),
        )

        categories = [tag["term"] for tag in entry.get("tags", [])]
        primary_category = entry.get("arxiv_primary_category", {}).get(
            "term", categories[0] if categories else ""
        )

        return Paper(
            arxiv_id=arxiv_id,
            title=" ".join(entry.title.split()),
            abstract=" ".join(entry.summary.split()),
            authors=[author.name for author in entry.get("authors", [])],
            categories=categories,
            primary_category=primary_category,
            published=_parse_time(entry.published_parsed),
            updated=_parse_time(entry.updated_parsed),
            pdf_url=pdf_url,
            abs_url=entry.id,
            comment=entry.get("arxiv_comment"),
            journal_ref=entry.get("arxiv_journal_ref"),
        )


def _parse_time(struct_time) -> "datetime":
    from datetime import datetime

    return datetime(*struct_time[:6])