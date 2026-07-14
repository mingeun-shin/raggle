import indexing.chunk_store as chunk_store
import indexing.dense_index as dense_index
import indexing.sparse_index as sparse_index
from indexing.embedder import embed_texts
from ingestion.chunking import chunk_document
from ingestion.loaders import load_directory
from settings import settings

import indexing.chunk_store as chunk_store
from generation.answer import Answer, generate_answer
from retrieval.rerank import rerank
from retrieval.search import hybrid_search
from settings import settings


def ingest(corpus_dir: str) -> dict:
    docs = load_directory(corpus_dir)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    # full rebuild: idempotent, no partial-index states to reason about
    chunk_store.clear()
    dense_index.clear()

    chunk_store.save_chunks(all_chunks)
    vectors = embed_texts([c.text for c in all_chunks])
    dense_index.upsert([c.chunk_id for c in all_chunks], vectors)
    sparse_index.build()

    return {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "config_hash": settings.config_hash(),
    }

def retrieved_ids(question: str) -> list[str]:
    candidates = hybrid_search(question)
    if not candidates:
        return []
    chunks = chunk_store.get_chunks(candidates)
    return [c.chunk_id for c in rerank(question, chunks, settings.rerank_top_m)]


def query(question: str) -> Answer:
    ids = retrieved_ids(question)
    if not ids:
        return Answer(text="No documents have been ingested.", citations={}, contexts=[])
    return generate_answer(question, chunk_store.get_chunks(ids))