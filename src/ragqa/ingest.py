from __future__ import annotations

from pathlib import Path

from .chunking import simple_char_chunks
from .config import cfg
from .embedder import Embedder
from .vectorstore import VectorStore


def load_docs(docs_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in [".md", ".txt"]:
            continue
        text = p.read_text(encoding="utf-8")
        docs.append((str(p.relative_to(docs_dir)), text))
    return docs


def main() -> None:
    cfg.index_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(cfg.docs_dir)
    if not docs:
        raise SystemExit(f"No docs found in {cfg.docs_dir}. Put .md/.txt files there.")

    embedder = Embedder()
    all_chunks = []
    for doc_id, text in docs:
        all_chunks.extend(
            simple_char_chunks(doc_id, text, cfg.chunk_size, cfg.chunk_overlap)
        )

    texts = [c.text for c in all_chunks]
    embs = embedder.embed_texts(texts)

    vs = VectorStore(dim=embs.shape[1])
    vs.add(embs, all_chunks)
    vs.save(cfg.faiss_index_path, cfg.meta_path)

    print(
        f"Indexed docs={len(docs)} chunks={len(all_chunks)} -> {cfg.faiss_index_path}"
    )


if __name__ == "__main__":
    main()
