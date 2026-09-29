"""Generate a synthesized, cited answer to a question using the top-ranked papers.

This is the "generation" half of RAG: retrieval (calli.retrieval.ask) finds
the most relevant papers, this module turns them into an actual answer
grounded in what those papers say, rather than just a ranked list.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

# Cheap and fast - good for iterating on the prompt/pipeline without worrying
# about cost. Swap to "claude-sonnet-5" once you're tuning answer quality
# itself rather than debugging the pipeline.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a research assistant helping someone understand a topic \
by synthesizing findings from academic papers. You will be given a question and a \
list of candidate papers (title, abstract, arXiv id), ranked by relevance.

Write a concise answer (3-6 sentences) that directly addresses the question, based \
only on what these abstracts actually say. After each claim, cite which paper(s) \
support it using their arXiv id in brackets, e.g. [2005.11401v4]. If the papers \
don't fully answer the question, say so plainly rather than filling gaps with \
outside knowledge. Do not invent findings that aren't present in the abstracts."""


def build_user_prompt(question: str, papers: list[dict]) -> str:
    """Build the prompt body listing the question and candidate papers.

    Split out as its own function (rather than inlined in synthesize_answer)
    so it can be unit tested without needing a real API call.
    """
    lines = [f"Question: {question}", "", "Candidate papers:"]
    for i, paper in enumerate(papers, start=1):
        lines.append(
            f"\n{i}. [{paper['arxiv_id']}] {paper['title']}\n   Abstract: {paper['abstract']}"
        )
    return "\n".join(lines)


def synthesize_answer(
    question: str,
    papers: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
) -> str:
    """Given a question and ranked papers (as returned by calli.retrieval.ask.ask),
    return a synthesized answer citing which papers support each claim.

    Requires ANTHROPIC_API_KEY to be set in the environment.
    """
    if not papers:
        return "No relevant papers were found to answer this question."

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, papers)}],
    )
    return response.content[0].text