from __future__ import annotations

import sys

from .schemas import AnswerResult

# serviceからロジックを呼ぶ
from .service import answer_question


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m ragqa.ask "質問文"')

    question = sys.argv[1]

    try:
        # ロジック呼び出し：たったの1行！
        result: AnswerResult = answer_question(question)
    except Exception as e:
        print(f"Error: {e}")
        return

    # === 表示ロジック ===
    print("==== Retrieved ====")
    for s in result.sources:
        print(f"- {s.doc_id}#{s.chunk_id} score={s.score:.3f}")

    print("\n==== Answer ====")
    print(result.answer)

    print("\n==== Evidence Check ====")
    if result.verification.supported_claims:
        print("- supported_claims:")
        for c in result.verification.supported_claims[:10]:
            print(f"  - {c}")

    print(
        f"- verdict: {result.verification.verdict} (confidence={result.verification.confidence})"
    )

    if result.verification.missing_points:
        print("- missing_points:")
        for p in result.verification.missing_points[:20]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
