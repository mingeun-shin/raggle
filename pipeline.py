import indexing.chunk_store as chunk_store
import indexing.dense_index as dense_index
import indexing.sparse_index as sparse_index
from indexing.embedder import embed_texts
from ingestion.chunking import chunk_document
from ingestion.loaders import load_directory
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