# spec-rag-qa

業務仕様・設計書（Markdown/Text）を対象に、RAGでQAする最小プロジェクト。

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m ragqa.ingest
python -m ragqa.ask "この仕様の例外条件は？"
