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

    # === BM25インデックス構築 ===
    from .bm25_store import BM25Store

    bm25 = BM25Store(b=cfg.bm25_b, k1=cfg.bm25_k1)
    bm25.build(all_chunks)
    bm25.save(cfg.bm25_index_path, cfg.bm25_postings_path)
    print(f"BM25 indexed: vocab={len(bm25.df)} chunks={bm25.N} avgdl={bm25.avgdl:.1f}")

    # === manifest.json 生成 ===
    import datetime
    import hashlib
    import json

    doc_hashes = {}
    for doc_id, text in docs:
        doc_hashes[doc_id] = hashlib.sha256(text.encode()).hexdigest()[:16]

    manifest = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "doc_hashes": doc_hashes,
        "embedding_config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "bm25_config": {"b": cfg.bm25_b, "k1": cfg.bm25_k1},
        "integrity": {"total_chunks": len(all_chunks), "vocab_size": len(bm25.df)},
    }
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest saved: {cfg.manifest_path}")

    print(
        f"Indexed docs={len(docs)} chunks={len(all_chunks)} -> {cfg.faiss_index_path}"
    )


if __name__ == "__main__":
    main()
