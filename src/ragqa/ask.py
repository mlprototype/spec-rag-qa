from __future__ import annotations

import sys

from .config import cfg
from .embedder import Embedder
from .llm import run_llm
from .prompt import build_prompt
from .vectorstore import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m ragqa.ask "質問文"')

    question = sys.argv[1]

    vs = VectorStore.load(cfg.faiss_index_path, cfg.meta_path)
    embedder = Embedder()
    q = embedder.embed_query(question)

    hits = vs.search(q, cfg.top_k)
    contexts = [h["text"] for h in hits]

    prompt = build_prompt(question, contexts)
    ans = run_llm(prompt, contexts)

    print("==== Retrieved ====")
    for h in hits:
        print(f"- {h['doc_id']}#{h['chunk_id']} score={h['score']:.3f}")
    print("\n==== Answer ====")
    print(ans)


if __name__ == "__main__":
    main()
