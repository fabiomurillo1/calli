"""Ingest papers from arXiv into data/raw as JSON Lines.

Examples:
    python scripts/ingest.py "cat:cs.CL AND abs:retrieval" --max-results 50
    python scripts/ingest.py "au:Bengio_Y" --max-results 10 --output data/raw/bengio.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from calli.ingestion.arxiv_client import ArxivClient

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers from arXiv")
    parser.add_argument(
        "query",
        help="arXiv search query, e.g. 'cat:cs.CL AND abs:\"retrieval augmented generation\"'",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/raw/ingested_papers.jsonl)",
    )
    args = parser.parse_args()

    client = ArxivClient()
    papers = client.search(args.query, max_results=args.max_results)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else DATA_RAW / "ingested_papers.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")

    print(f"Ingested {len(papers)} papers -> {output_path}")


if __name__ == "__main__":
    main()