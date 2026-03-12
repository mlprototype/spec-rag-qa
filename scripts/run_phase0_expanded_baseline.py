from __future__ import annotations

import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragqa.chunking import markdown_header_chunks
from ragqa.embedder import Embedder
from ragqa.vectorstore import VectorStore

SEED = 20260223
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

BASE_DIR = Path("data/phase0_expanded")
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = Path("data/index/phase0_expanded")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

EVAL_DIR = Path("data/eval")
CASES_PATH = EVAL_DIR / "ground_truth_phase0_expanded.json"
CORPUS_LIST_PATH = EVAL_DIR / "phase0_expanded_corpus_list.json"
REPORT_JSON_PATH = EVAL_DIR / "phase0_vector_baseline_expanded.json"
REPORT_MD_PATH = EVAL_DIR / "phase0_vector_baseline_expanded_report.md"
LEGACY_BASELINE_PATH = EVAL_DIR / "phase0_vector_baseline.json"


@dataclass(frozen=True)
class DocSpec:
    filename: str
    title: str
    category: str
    is_noise: bool
    key_terms: list[str]
    mandatory_facts: list[str]


DOC_SPECS: list[DocSpec] = [
    DocSpec(
        filename="sample_spec_v2.md",
        title="会員登録仕様書 v2",
        category="spec",
        is_noise=False,
        key_terms=["会員登録", "メール重複", "パスワード最小長", "検証エラー", "競合制御"],
        mandatory_facts=[
            "会員登録の正式なAPIは POST /api/signup である。",
            "登録済みメールアドレスを使用した場合は HTTP 409 Conflict を返す。",
            "パスワードの最低文字数は 8 文字である。",
            "英数字混在は現行では推奨であり、将来要件として必須化を検討している。",
        ],
    ),
    DocSpec(
        filename="auth_design_detail.md",
        title="認証基盤 詳細設計",
        category="design",
        is_noise=False,
        key_terms=["認証基盤", "パスワードポリシー", "機能フラグ", "例外分岐", "運用診断"],
        mandatory_facts=[
            "現行リリースでは英数字混在は必須ではなく推奨扱いとする。",
            "移行完了後に機能フラグで厳格な構成要件を有効化できる。",
            "パスワード再設定の例外経路はインシデント文脈と運用メモを残す。",
        ],
    ),
    DocSpec(
        filename="error_code_reference.md",
        title="エラーコード リファレンス",
        category="reference",
        is_noise=False,
        key_terms=["ステータスコード", "バリデーション", "認可", "競合", "意味エラー"],
        mandatory_facts=[
            "HTTP 400 は入力バリデーション失敗を表す。",
            "HTTP 401 は認証失敗または未認証を表す。",
            "HTTP 403 は権限不足を表す。",
            "HTTP 404 は対象リソース未検出を表す。",
            "HTTP 409 は会員登録時のメール重複競合を表す。",
            "HTTP 422 は意味的に不正な要求を表す。",
        ],
    ),
    DocSpec(
        filename="retention_policy.md",
        title="個人データ保持ポリシー",
        category="policy",
        is_noise=False,
        key_terms=["保持期間", "パージ猶予", "削除フロー", "法的保留", "監査証跡"],
        mandatory_facts=[
            "退会後のプロフィール情報は法的保留がない場合 30 日後に削除する。",
            "セキュリティ監査ログの保持期間は 180 日である。",
            "停止中アカウントは削除承認が完了するまでパージ対象にしない。",
        ],
    ),
    DocSpec(
        filename="password_policy_faq.md",
        title="パスワードポリシー FAQ",
        category="faq",
        is_noise=False,
        key_terms=["FAQ", "強度", "推奨", "必須要件", "利用者ガイド"],
        mandatory_facts=[
            "利用者は 8 文字以上のパスワードを設定しなければならない。",
            "英数字混在は強く推奨するが、現時点では必須ではない。",
            "ロードマップ上は推奨から必須への変更を検討している。",
        ],
    ),
    DocSpec(
        filename="session_management_design.md",
        title="セッション管理 設計書",
        category="design",
        is_noise=False,
        key_terms=["セッションタイムアウト", "アイドル期限", "トークン寿命", "リスク軽減", "再認証"],
        mandatory_facts=[
            "対話セッションは 30 分の無操作で期限切れとする。",
            "標準利用者の絶対セッション寿命は 24 時間とする。",
            "高リスク検知時またはアイドル期限切れ時は再認証を要求する。",
        ],
    ),
    DocSpec(
        filename="account_lifecycle_spec.md",
        title="アカウントライフサイクル仕様",
        category="spec",
        is_noise=False,
        key_terms=["状態遷移", "停止フロー", "削除承認", "最終パージ", "運用引継ぎ"],
        mandatory_facts=[
            "アカウント状態は active, suspended, deleted, purged で定義する。",
            "停止は可逆であり、削除は明示的承認を必要とする。",
            "削除後 30 日経過かつ法的保留なしで purged へ遷移する。",
        ],
    ),
    DocSpec(
        filename="api_contract_signup.md",
        title="会員登録 API 契約",
        category="contract",
        is_noise=False,
        key_terms=["API契約", "リクエストスキーマ", "レスポンス定義", "エンドポイント", "エラーマッピング"],
        mandatory_facts=[
            "POST /api/signup は email, password, locale を受け付ける。",
            "メール重複時は HTTP 409 とエラーコード DUPLICATE_EMAIL を返す。",
            "入力不正時は HTTP 400 と項目別エラー詳細を返す。",
        ],
    ),
    DocSpec(
        filename="incident_runbook_auth.md",
        title="認証障害ランブック",
        category="runbook",
        is_noise=False,
        key_terms=["障害トリアージ", "エラースパイク", "エスカレーション", "当番対応", "封じ込め"],
        mandatory_facts=[
            "会員登録で HTTP 409 が急増した場合は認証基盤のトリアージを開始する。",
            "競合率が 15 分以上ベースライン超過した場合は当番エスカレーションを行う。",
            "サポート向け案内とロールバック手順を同時に参照する。",
        ],
    ),
    DocSpec(
        filename="support_playbook_identity.md",
        title="認証サポート プレイブック",
        category="operations",
        is_noise=False,
        key_terms=["サポート手順", "メール重複", "顧客案内", "アカウントロック", "引継ぎ"],
        mandatory_facts=[
            "会員登録競合時の初動はメール重複確認である。",
            "サポート担当はパスワード再設定案内前にアカウント状態を確認する。",
            "アカウントロック対応では所定テンプレートを利用する。",
        ],
    ),
    DocSpec(
        filename="data_classification_policy.md",
        title="データ分類と保持マッピング",
        category="policy",
        is_noise=False,
        key_terms=["データ分類", "制限情報", "保持マッピング", "コンプライアンス", "統制"],
        mandatory_facts=[
            "データ分類は public, internal, restricted の 3 区分で定義する。",
            "restricted 情報は厳格な保持制御を適用する。",
            "保持マッピングは法務・監査・セキュリティ義務に紐付く。",
        ],
    ),
    DocSpec(
        filename="audit_logging_standard.md",
        title="監査ログ標準",
        category="standard",
        is_noise=False,
        key_terms=["監査ログ", "USER_ID", "追跡性", "保持期間", "フォレンジック"],
        mandatory_facts=[
            "認証・認可ログには USER_ID を必ず含める。",
            "認証ログの保持期間は 180 日とする。",
            "ログスキーマには request_id, actor_type, USER_ID, action_result を含める。",
        ],
    ),
    DocSpec(
        filename="unrelated_marketing_doc.md",
        title="マーケティング四半期レポート",
        category="noise",
        is_noise=True,
        key_terms=["キャンペーン", "コンバージョン", "ブランド訴求", "ローンチ文言", "申込導線"],
        mandatory_facts=[
            "本資料は施策評価を目的とし、認証運用の正式根拠にはならない。",
            "会員登録という語は含むが、API定義やエラーコード規約は扱わない。",
        ],
    ),
    DocSpec(
        filename="unrelated_company_history.md",
        title="企業沿革アーカイブ",
        category="noise",
        is_noise=True,
        key_terms=["沿革", "創業年", "拠点拡大", "文化施策", "ブランド史"],
        mandatory_facts=[
            "利用者増加推移は記載するが、認証実装や保持ポリシーは記載しない。",
            "統制用語を含むが運用判断の根拠としては無効である。",
        ],
    ),
    DocSpec(
        filename="unrelated_travel_policy.md",
        title="出張・経費規程",
        category="noise",
        is_noise=True,
        key_terms=["出張申請", "経費精算", "承認フロー", "保存年限", "規程ポータル"],
        mandatory_facts=[
            "経費記録の保持は財務監査向けで、認証設計とは関係しない。",
            "保持・承認という語を含むが認証運用の判断材料にはならない。",
        ],
    ),
]


def _token_count(text: str) -> int:
    tokens = re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9_./-]+", text)
    return len(tokens)


def _paragraph(rng: random.Random, title: str, terms: list[str], section: str) -> str:
    t1, t2, t3 = rng.sample(terms, 3)
    templates = [
        f"{section}では「{t1}」と「{t2}」の記述を近い文脈で配置し、審査者が「{t3}」を根拠として判断する際に取り違えないよう境界条件を明示する。",
        f"{title}の利用者は「{t1}」と「{t2}」を同じ概念として誤読しやすいため、本節では例外時の責任分界と証跡の残し方を「{t3}」に紐付けて反復説明する。",
        f"実務文書では言い回しが重複するため、本節は「{t1}」「{t2}」「{t3}」を似た構文で複数回記述し、検索時に曖昧性が発生する状況を意図的に再現する。",
        f"本節は平常時と例外時を比較しながら「{t1}」の条件を整理し、「{t2}」が成立する前提と「{t3}」の確認手順を段階的に列挙する。",
    ]
    return " ".join(rng.sample(templates, 3))


def _build_doc(spec: DocSpec, rng: random.Random) -> str:
    sections = [
        "目的",
        "適用範囲",
        "機能メモ",
        "例外処理",
        "運用ガイダンス",
        "FAQ",
        "変更管理",
    ]

    lines: list[str] = [f"# {spec.title}", ""]
    lines.append("この文書は仕様書RAGの検索評価向けに作成した合成コーパスです。")
    lines.append("")
    lines.append("## 主要事実")
    for fact in spec.mandatory_facts:
        lines.append(f"- {fact}")
    lines.append("")

    for section in sections:
        lines.append(f"## {section}")
        lines.append(_paragraph(rng, spec.title, spec.key_terms, section))
        lines.append(_paragraph(rng, spec.title, spec.key_terms, section))
        lines.append("")

    while _token_count("\n".join(lines)) < 320:
        lines.append("## 補足メモ")
        lines.append(_paragraph(rng, spec.title, spec.key_terms, "補足メモ"))
        lines.append("")

    text = "\n".join(lines).strip() + "\n"
    tokens = re.findall(r"[^\s]+|\n", text)
    if _token_count(text) > 980:
        current = 0
        out: list[str] = []
        for tk in tokens:
            if tk == "\n":
                out.append("\n")
                continue
            current += 1
            if current > 980:
                break
            out.append(tk + " ")
        text = "".join(out).replace(" \n", "\n")
    return text


def _build_cases() -> list[dict]:
    evaluable = [
        {
            "id": "exp-001",
            "type": "factual_basic",
            "question": "登録済みメールアドレスで会員登録した場合のステータスコードは何ですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答はメール重複時に 409 を返すことを含む。",
            "expected_sources": ["sample_spec_v2.md", "error_code_reference.md"],
        },
        {
            "id": "exp-002",
            "type": "paraphrase",
            "question": "重複アカウント登録時の応答コードを教えてください。",
            "expected_verdict": "sufficient",
            "assertion": "回答は重複登録と 409 の対応を述べる。",
            "expected_sources": ["api_contract_signup.md", "error_code_reference.md"],
        },
        {
            "id": "exp-003",
            "type": "priority_conflict",
            "question": "英数字混在のパスワードは今すぐ必須ですか、それとも推奨ですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は現時点で推奨であり必須ではないことを示す。",
            "expected_sources": [
                "sample_spec_v2.md",
                "auth_design_detail.md",
                "password_policy_faq.md",
            ],
        },
        {
            "id": "exp-004",
            "type": "factual_basic",
            "question": "signup のエンドポイント定義はどの文書にありますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は POST /api/signup を含む。",
            "expected_sources": ["api_contract_signup.md", "sample_spec_v2.md"],
        },
        {
            "id": "exp-005",
            "type": "cross_doc",
            "question": "アカウント削除後、プロフィールデータは何日後にパージされますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は 30 日と削除状態を言及する。",
            "expected_sources": ["retention_policy.md", "account_lifecycle_spec.md"],
        },
        {
            "id": "exp-006",
            "type": "factual_basic",
            "question": "USER_ID を監査ログに必須とする記載はどこですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は USER_ID 必須記載に言及する。",
            "expected_sources": ["audit_logging_standard.md"],
        },
        {
            "id": "exp-007",
            "type": "cross_doc",
            "question": "409エラーが継続したとき、サポートの最初の対応は何ですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は重複メール確認とトリアージを結びつける。",
            "expected_sources": ["support_playbook_identity.md", "incident_runbook_auth.md"],
        },
        {
            "id": "exp-008",
            "type": "factual_basic",
            "question": "対話セッションのアイドルタイムアウトは何分ですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は 30 分の無操作タイムアウトを含む。",
            "expected_sources": ["session_management_design.md"],
        },
        {
            "id": "exp-009",
            "type": "paraphrase",
            "question": "パスワード再設定の例外経路はどこに書かれていますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は設計資料とサポート資料への参照を含む。",
            "expected_sources": ["auth_design_detail.md", "support_playbook_identity.md"],
        },
        {
            "id": "exp-010",
            "type": "factual_basic",
            "question": "データ分類と保持マッピングを定義している文書はどれですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は public/internal/restricted の区分に言及する。",
            "expected_sources": ["data_classification_policy.md"],
        },
        {
            "id": "exp-011",
            "type": "misleading",
            "question": "重複メールは 422 ですか、それとも 409 ですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は 409 と 422 の違いを明確に区別する。",
            "expected_sources": ["error_code_reference.md", "sample_spec_v2.md"],
        },
        {
            "id": "exp-012",
            "type": "cross_doc",
            "question": "最終パージ前に存在するアカウント状態を教えてください。",
            "expected_verdict": "sufficient",
            "assertion": "回答は active, suspended, deleted, purged に言及する。",
            "expected_sources": ["account_lifecycle_spec.md"],
        },
        {
            "id": "exp-013",
            "type": "factual_basic",
            "question": "認証インシデントのトリアージ手順はどこにありますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は認証障害ランブックを参照する。",
            "expected_sources": ["incident_runbook_auth.md"],
        },
        {
            "id": "exp-014",
            "type": "paraphrase",
            "question": "パスワード複雑性は強制要件ではなくガイダンスだと書かれているのはどれですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は推奨扱いであることを含む。",
            "expected_sources": ["password_policy_faq.md", "auth_design_detail.md"],
        },
        {
            "id": "exp-015",
            "type": "factual_basic",
            "question": "signup リクエストの項目（email/password/locale）はどこに定義されていますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は payload 項目定義を含む。",
            "expected_sources": ["api_contract_signup.md", "sample_spec_v2.md"],
        },
        {
            "id": "exp-016",
            "type": "cross_doc",
            "question": "セキュリティ監査ログの保持期間を定義している資料はどれですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は監査ログ標準と保持ポリシーを結びつける。",
            "expected_sources": ["audit_logging_standard.md", "retention_policy.md"],
        },
        {
            "id": "exp-017",
            "type": "factual_basic",
            "question": "メール重複と HTTP 409 の対応を明示している箇所はどこですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は重複メールと 409 を明示する。",
            "expected_sources": ["sample_spec_v2.md", "error_code_reference.md"],
        },
        {
            "id": "exp-018",
            "type": "factual_basic",
            "question": "アカウントロック時の連絡テンプレートについて書かれている資料は？",
            "expected_verdict": "sufficient",
            "assertion": "回答はサポートプレイブックを参照する。",
            "expected_sources": ["support_playbook_identity.md"],
        },
        {
            "id": "exp-019",
            "type": "ambiguity",
            "question": "停止と削除の分岐判断フローはどこに記載されていますか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は停止可逆・削除承認の条件を含む。",
            "expected_sources": ["account_lifecycle_spec.md", "retention_policy.md"],
        },
        {
            "id": "exp-020",
            "type": "cross_doc",
            "question": "競合処理とエスカレーショントリガーの両方に触れている資料はどれですか？",
            "expected_verdict": "sufficient",
            "assertion": "回答は競合処理と障害エスカレーションを接続する。",
            "expected_sources": ["incident_runbook_auth.md", "error_code_reference.md"],
        },
    ]

    no_source = [
        {
            "id": "exp-021",
            "type": "omission_detection",
            "question": "アバター画像の最大アップロード容量は何MBですか？",
            "expected_verdict": "insufficient",
            "assertion": "回答は資料に定義がないことを述べる。",
            "expected_sources": [],
        },
        {
            "id": "exp-022",
            "type": "omission_detection",
            "question": "Appleソーシャルログインは標準で有効ですか？",
            "expected_verdict": "insufficient",
            "assertion": "回答はソーシャルログイン仕様を捏造しない。",
            "expected_sources": [],
        },
        {
            "id": "exp-023",
            "type": "opinion_guard",
            "question": "ダークモードのプライマリボタンデザイントークンは何ですか？",
            "expected_verdict": "insufficient",
            "assertion": "回答はUIトークン仕様が存在しないことを明示する。",
            "expected_sources": [],
        },
        {
            "id": "exp-024",
            "type": "omission_detection",
            "question": "管理者ロールに生体認証は必須ですか？",
            "expected_verdict": "insufficient",
            "assertion": "回答は生体認証必須を推測で断定しない。",
            "expected_sources": [],
        },
        {
            "id": "exp-025",
            "type": "omission_detection",
            "question": "電話サポートの応答SLAは何分ですか？",
            "expected_verdict": "insufficient",
            "assertion": "回答は分単位SLAが未定義であると述べる。",
            "expected_sources": [],
        },
    ]

    out = evaluable + no_source
    for item in out:
        item["sources"] = item["expected_sources"]
    return out


def _write_dataset(seed: int) -> list[dict]:
    rng = random.Random(seed)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    corpus_rows: list[dict] = []
    for spec in DOC_SPECS:
        text = _build_doc(spec, rng)
        (DOCS_DIR / spec.filename).write_text(text, encoding="utf-8")
        corpus_rows.append(
            {
                "filename": spec.filename,
                "title": spec.title,
                "category": spec.category,
                "is_noise": spec.is_noise,
                "word_count": _token_count(text),
            }
        )

    cases = _build_cases()
    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    CORPUS_LIST_PATH.write_text(
        json.dumps(
            {
                "seed": seed,
                "docs_dir": str(DOCS_DIR),
                "doc_count": len(corpus_rows),
                "docs": corpus_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return cases


def _load_docs() -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in sorted(DOCS_DIR.rglob("*.md")):
        docs.append((str(p.relative_to(DOCS_DIR)), p.read_text(encoding="utf-8")))
    return docs


def _build_vector_index() -> None:
    docs = _load_docs()
    all_chunks = []
    for doc_id, text in docs:
        all_chunks.extend(markdown_header_chunks(doc_id, text))
    texts = [c.text for c in all_chunks]
    embedder = Embedder(EMBEDDING_MODEL)
    embs = embedder.embed_texts(texts)
    vs = VectorStore(dim=embs.shape[1])
    vs.add(embs, all_chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save(INDEX_PATH, META_PATH)


def _run_baseline(cases: list[dict]) -> dict:
    vs = VectorStore.load(INDEX_PATH, META_PATH)
    embedder = Embedder(EMBEDDING_MODEL)

    rows = []
    latencies = []
    for case in cases:
        t0 = time.perf_counter()
        q_emb = embedder.embed_query(case["question"])
        hits = vs.search(q_emb, TOP_K)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved = [h["doc_id"] for h in hits]
        expected = case.get("expected_sources", [])
        first_rank = None
        if expected:
            expected_set = set(expected)
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in expected_set:
                    first_rank = rank
                    break

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "type": case["type"],
                "expected_sources": expected,
                "retrieved_doc_ids_top_k": retrieved,
                "first_relevant_rank": first_rank,
                "latency_ms": round(latency_ms, 3),
            }
        )

    evaluable = [r for r in rows if r["expected_sources"]]
    hit_at_1 = sum(1 for r in evaluable if r["first_relevant_rank"] == 1)
    hit_at_5 = sum(1 for r in evaluable if r["first_relevant_rank"] is not None)
    recall_at_1 = hit_at_1 / len(evaluable) if evaluable else 0.0
    recall_at_5 = hit_at_5 / len(evaluable) if evaluable else 0.0
    mrr = (
        sum(
            1.0 / r["first_relevant_rank"]
            for r in evaluable
            if r["first_relevant_rank"] is not None
        )
        / len(evaluable)
        if evaluable
        else 0.0
    )
    failure_rate = 1.0 - recall_at_5 if evaluable else 0.0

    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0

    return {
        "phase": "Phase 0 Expanded",
        "mode": "vector_only",
        "seed": SEED,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "metric_definition": {
            "recall_at_1": "expected_sources があるケースで top1 に正解docが含まれる割合",
            "recall_at_5": "expected_sources があるケースで top5 に正解docが1件以上含まれる割合",
            "mrr": "expected_sources があるケースの reciprocal rank 平均（top_k=5）",
            "failure_rate": "1 - recall_at_5（retrieval miss rate）",
            "p95_latency_ms": "1クエリあたり retrieval（embed_query + faiss search）の95パーセンタイル",
            "p50_latency_ms": "1クエリあたり retrieval（embed_query + faiss search）の50パーセンタイル",
        },
        "summary": {
            "total_cases": len(rows),
            "retrieval_evaluable_cases": len(evaluable),
            "no_source_cases": len(rows) - len(evaluable),
            "recall_at_1": round(recall_at_1, 6),
            "recall_at_5": round(recall_at_5, 6),
            "mrr": round(mrr, 6),
            "failure_rate": round(failure_rate, 6),
            "p95_latency_ms": round(p95, 3),
            "p50_latency_ms": round(p50, 3),
        },
        "cases": rows,
    }


def _attach_comparison(out: dict) -> dict:
    if not LEGACY_BASELINE_PATH.exists():
        out["comparison_to_previous_phase0"] = {"status": "legacy baseline not found"}
        return out

    legacy = json.loads(LEGACY_BASELINE_PATH.read_text(encoding="utf-8"))
    old = legacy.get("summary", {})
    new = out["summary"]

    def delta(key: str):
        if key not in old or key not in new:
            return None
        return round(float(new[key]) - float(old[key]), 6)

    out["comparison_to_previous_phase0"] = {
        "legacy_file": str(LEGACY_BASELINE_PATH),
        "old_retrieval_evaluable_cases": old.get("retrieval_evaluable_cases"),
        "new_retrieval_evaluable_cases": new.get("retrieval_evaluable_cases"),
        "delta_recall_at_5": delta("recall_at_5"),
        "delta_mrr": delta("mrr"),
        "delta_failure_rate": delta("failure_rate"),
        "delta_p95_latency_ms": delta("p95_latency_ms"),
    }
    return out


def _write_report(out: dict) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    c = out.get("comparison_to_previous_phase0", {})
    ratio = out["summary"]["no_source_cases"] / out["summary"]["total_cases"]
    md = f"""# Phase 0 拡張ベースラインレポート（Vector-only）

- seed: `{SEED}`
- embedding_model: `{EMBEDDING_MODEL}`
- top_k: `{TOP_K}`
- total_cases: `{out['summary']['total_cases']}`
- retrieval_evaluable_cases: `{out['summary']['retrieval_evaluable_cases']}`
- no_source_cases: `{out['summary']['no_source_cases']}` ({ratio:.1%})

## 指標

- Recall@1: {out['summary']['recall_at_1']:.4f}
- Recall@5: {out['summary']['recall_at_5']:.4f}
- MRR: {out['summary']['mrr']:.4f}
- Failure Rate: {out['summary']['failure_rate']:.4f}
- p95 Latency (ms): {out['summary']['p95_latency_ms']:.3f}
- p50 Latency (ms): {out['summary']['p50_latency_ms']:.3f}

## 旧Phase0との差分

- 旧evaluable cases: {c.get('old_retrieval_evaluable_cases')}
- 新evaluable cases: {c.get('new_retrieval_evaluable_cases')}
- Delta Recall@5: {c.get('delta_recall_at_5')}
- Delta MRR: {c.get('delta_mrr')}
- Delta Failure Rate: {c.get('delta_failure_rate')}
- Delta p95 Latency (ms): {c.get('delta_p95_latency_ms')}

## 自己評価

- コーパス規模（10件以上）: {'PASS' if len(DOC_SPECS) >= 10 else 'FAIL'} ({len(DOC_SPECS)} docs)
- retrieval_evaluable_cases >= 20: {'PASS' if out['summary']['retrieval_evaluable_cases'] >= 20 else 'FAIL'}
- no-source 比率が約20%: {'PASS' if 0.15 <= ratio <= 0.25 else 'WARN'}
"""
    REPORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cases = _write_dataset(SEED)
    _build_vector_index()
    out = _run_baseline(cases)
    out = _attach_comparison(out)
    _write_report(out)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote: {CASES_PATH}")
    print(f"Wrote: {CORPUS_LIST_PATH}")
    print(f"Wrote: {REPORT_JSON_PATH}")
    print(f"Wrote: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
