from __future__ import annotations

from pathlib import Path

from .chunking import markdown_header_chunks, simple_char_chunks
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
        # doc_id は文字列なので、末尾チェックで判定するのが確実です
        if doc_id.lower().endswith(".md"):
            print(f"Chunking Markdown: {doc_id}")
            all_chunks.extend(markdown_header_chunks(doc_id, text))
        else:
            # .txt などの場合は従来の文字数分割（設定値を使用）
            print(f"Chunking Text: {doc_id}")
            all_chunks.extend(
                simple_char_chunks(doc_id, text, cfg.chunk_size, cfg.chunk_overlap)
            )

    texts = [c.text for c in all_chunks]
    # データが空の場合のエラーハンドリング（念のため）
    if not texts:
        print("No chunks generated. Check your docs.")
        return

    embs = embedder.embed_texts(texts)

    vs = VectorStore(dim=embs.shape[1])
    vs.add(embs, all_chunks)
    vs.save(cfg.faiss_index_path, cfg.meta_path)

    print(
        f"Indexed docs={len(docs)} chunks={len(all_chunks)} -> {cfg.faiss_index_path}"
    )


if __name__ == "__main__":
    main()
