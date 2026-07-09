import hashlib
import os
from dataclasses import dataclass

from pypdf import PdfReader

from ingestion.normalize import normalize

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    doc_id: str
    text: str
    source_path: str
    metadata: dict


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_file(path: str) -> Document:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported file type: {path}")

    raw = _read_pdf(path) if ext == ".pdf" else _read_text(path)
    text = normalize(raw)
    # id from relative path, stable across re-ingestion of the same corpus layout
    doc_id = hashlib.sha256(os.path.basename(path).encode()).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=path,
        metadata={"filename": os.path.basename(path), "ext": ext},
    )


def load_directory(path: str) -> list[Document]:
    docs = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            full = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() in SUPPORTED:
                docs.append(load_file(full))
    if not docs:
        raise ValueError(f"no supported files found in {path}")
    return docs