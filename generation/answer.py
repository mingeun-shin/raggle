import re
from dataclasses import dataclass

from generation.llm import complete
from generation.prompts import build_answer_prompt
from ingestion.chunking import Chunk


@dataclass
class Answer:
    text: str
    citations: dict[int, str]
    contexts: list[Chunk]


def generate_answer(query: str, chunks: list[Chunk]) -> Answer:
    prompt = build_answer_prompt(query, [c.text for c in chunks])
    text = complete(prompt)

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    citations = {
        n: chunks[n - 1].chunk_id for n in sorted(cited) if 1 <= n <= len(chunks)
    }
    return Answer(text=text, citations=citations, contexts=chunks)