# Phase 6 Agent評価データセットとRunner

## 目的

`phase6_synthetic.json` は、Agentの業務意図ごとの経路選択、Tool利用、Citation、回答形式、予算、Task Successを決定論的に検証する公開可能な合成データセットです。質問から先に期待値を定義しており、Fixtureの実行結果へ合わせて期待routeを変更しません。

保存済みTraceは評価ロジックから分離されています。このため、Evaluatorを変更したときはAgentを再実行せず、同じTraceを再評価できます。Runnerの起動失敗や出力契約違反は `execution_errors` へ記録され、Task SuccessのFAILとは区別されます。

## ケース一覧

| Case ID | Category | Severity | 許容route | 必須Tool |
|:---|:---|:---|:---|:---|
| `agent-direct-greeting` | direct | low | direct | なし |
| `agent-direct-help` | direct | low | direct | なし |
| `agent-direct-scope` | direct | low | direct | なし |
| `agent-definition-session-timeout` | definition | medium | retrieval | hybrid_search |
| `agent-definition-password-policy` | definition | high | retrieval | hybrid_search |
| `agent-retrieval-error-409` | retrieval | high | retrieval | hybrid_search |
| `agent-retrieval-audit-user-id` | retrieval | high | retrieval | hybrid_search |
| `agent-complex-lockout-exceptions` | retrieval_complex | critical | retrieval | hybrid_search |
| `agent-complex-retention-conflict` | retrieval_complex | critical | retrieval | hybrid_search |
| `agent-complex-incident-procedure` | retrieval_complex | high | retrieval | hybrid_search |
| `agent-structured-orders-q1` | structured_query | medium | structured_query | structured_query_tool |
| `agent-structured-sales-q2` | structured_query | medium | structured_query | structured_query_tool |
| `agent-structured-inventory-average` | structured_query | medium | structured_query | structured_query_tool |
| `agent-structured-sales-top-q3` | structured_query | high | structured_query | structured_query_tool |
| `agent-compare-password-policies` | compare | high | compare | compare_documents |
| `agent-compare-retention-standards` | compare | high | compare | compare_documents |
| `agent-compare-api-versions` | compare | critical | compare | compare_documents |
| `agent-insufficient-future-roadmap` | insufficient_evidence | high | retrieval | hybrid_search |
| `agent-insufficient-unpublished-pricing` | insufficient_evidence | high | retrieval | hybrid_search |
| `agent-fallback-search-timeout` | fallback | critical | retrieval | hybrid_search |

カテゴリ分布は direct 3、definition 2、retrieval 2、retrieval_complex 3、structured_query 4、compare 3、insufficient_evidence 2、fallback 1です。definition/retrievalの合計は4件です。

## Runnerの利用方法

以下はリポジトリルートで実行します。Fixtureと保存TraceにはAPIキーが不要です。

```bash
PYTHONPATH=src python scripts/run_agent_evaluation.py --runner fixture
```

`fixture` は起動時に全Traceを読み込み、各ケースへ固定Traceのdeep copyを返します。

```bash
PYTHONPATH=src python scripts/run_agent_evaluation.py \
  --runner trace-file \
  --traces data/agent_eval/fixtures/phase6_synthetic_traces.json
```

`trace-file` は保存Traceを実行時に読み直します。`--traces` は単一JSONオブジェクト、JSON配列、または1ファイル1Traceのディレクトリを指定できます。

```bash
PYTHONPATH=src python scripts/run_agent_evaluation.py \
  --runner subprocess \
  --subprocess-command "python scripts/run_agent_trace.py" \
  --subprocess-cwd ../ai-agent-rag \
  --timeout-seconds 30 \
  --output /tmp/agent-evaluation.json
```

`subprocess` はシェルを経由せず、指定コマンドへ `--case-id`、`--question`、`--output` を追加します。ai-agent-rag#6の `timing.latency_ms`、nullableな `usage`、独立したCitation／Sourceを共通 `AgentRunTrace` として読み込みます。互換入力として `timing.total_latency_ms` とSource内の `citation_id` も正規化できます。

Runner timeout、終了コード非0、不正JSON、case_id不一致は、それぞれ専用例外となります。CLIは該当ケースを `execution_errors` へ記録し、ほかのケースの決定論的評価を継続します。終了コードは、全成功が0、評価FAILが1、事前検査または実行エラーが2です。

隣接 `ai-agent-rag` checkoutの `scripts/run_agent_trace.py` との実結合では、directの共通Routeと、Structured Queryケースの共通Route、`operation / target_metric / filters / target_dataset`、自然言語回答、Citationなしの契約を確認しています。retrieval、compareはモックSubprocessでAdapter経路を検証しています。実データを必要とするretrieval／compareの品質確認は、評価用コーパスの外部送信可否を確認した環境で実施します。

共通Routeは `direct / structured_query / retrieval / compare` の4種類です。内部RouteはTraceの `output.metadata.internal_route` に観測事実として残し、期待Routeには使用しません。Structured Queryの回答は自然言語として評価し、構造化データSource自体には回答内Citationが付与されない現行仕様に合わせてCitationを必須にしていません。Compareも自然言語回答を必須とし、`left / right / aspects` と既存Agentの4セクション「共通点」「相違点」「向いているケース（使い分けの指針）」「注意点」を評価します。セクションラベルはMarkdown見出し、番号付き項目、コロン付きラベルを認識しますが、本文中の部分文字列では合格しません。

## Report、品質Gate、Baseline

標準実行はGit管理されない `.artifacts/agent-quality/report.json` と `report.md` を生成します。Git管理する出力例は `data/agent_eval/reports/example.json` と `example.md` で、CI生成物とは分離しています。Reportには全体とcategory／severity別のTask Success、各決定論的指標、Route混同行列、latencyのaverage／p50／p95／max、Failure Typeとowner、`execution_error` を含みます。分母が0の指標はJSONで `null`、Markdownで `N/A` とし、100%には変換しません。

Gate定義は `config/agent_quality_gate.yml`、review済みBaselineは `data/agent_eval/baseline/agent_baseline.json` です。通常実行はBaselineを読み取るだけです。Baselineの置換は次の明示操作に限定され、いずれかの絶対GateまたはRunner実行に問題がある場合は更新しません。

```bash
PYTHONPATH=src python scripts/run_agent_evaluation.py \
  --runner fixture \
  --update-baseline
```

PR用GitHub ActionsはFixtureで同じGateを実行します。評価前に `.artifacts/agent-quality/` を削除・再作成し、今回生成したJSON、Markdown、実行ログだけを `if: always()` でartifact化します。実Agentは手動dispatchの別Jobに分離し、対象 `ai-agent-rag` revisionも明示して実行します。

## 事前検査

評価開始前に次を検査します。

- case idおよびTrace case_idの重複
- 必須Toolと禁止Toolの矛盾
- Tool引数schema/assertionから未定義Toolへの参照
- JSON Schema自体とローカル `$ref` の解決
- Fixtureの欠落、余剰、schema_version不一致、質問文不一致

## 合成データの既知の限界

- 会社名、文書、注文、売上などはすべて架空で、実案件の語彙・権限・機密区分を再現しません。
- Fixtureは意図的に決定論的で、LLMの揺らぎ、長い会話履歴、並行Tool実行、ストリーミングを含みません。
- latency、token、costは合成値であり、実環境のネットワーク、モデル、インフラSLOを代表しません。
- Tool catalogと引数schemaは代表例に限定され、実Agentの全Toolや認証・再試行挙動を網羅しません。
- Citationは構造化Traceと回答マーカーの整合性を検証しますが、引用内容がSourceの主張を意味的に支持するかは検証しません。
- カテゴリ比率は受入条件のカバレッジを目的としており、本番トラフィック分布ではありません。
- fallbackは保存Traceによる経路確認であり、実際の障害注入や外部サービス停止試験ではありません。

したがって、このデータセットは契約・回帰テストの土台です。本番導入判断には、匿名化した実質問、実行環境のTrace、人手判定、障害注入、セキュリティ評価を別途追加する必要があります。
