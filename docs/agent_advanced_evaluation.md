# Phase 6 高度Agent評価

## 目的と運用モード

Groundedness、Answer Semantic Consistency、Costは、外部Judge、複数回実行、価格表更新の影響を受けます。初期運用では既存の決定論的PR Gateへ含めず、`MONITOR ONLY`として傾向と失敗を可視化します。Route、Tool、CitationのStabilityも同じreportへ記録しますが、閾値による合否判定は行いません。

PRの `Offline Agent Quality Gate` は従来どおりFixtureだけで完結し、外部Judgeを呼びません。外部Judgeと実 `ai-agent-rag` を使う `Advanced Agent Monitoring` はGitHub Actionsの `workflow_dispatch` で `run_advanced_monitoring=true` を明示した場合だけ実行されます。JobはGitHub Environment `agent-evaluation`に所属し、機密データを含む評価はEnvironmentのrequired reviewerが承認した後だけ実行します。対象Agent revisionは `ai_agent_ref` で固定し、Agent用 `OPENAI_API_KEY` とJudge用 `AGENT_EVAL_JUDGE_API_KEY` は別secretとして扱います。

Judge endpointとmodelは自由入力ではありません。EnvironmentまたはRepositoryの次のVariablesで管理します。

- `AGENT_EVAL_JUDGE_URL`: 承認済みHTTPS endpoint
- `AGENT_EVAL_JUDGE_MODEL`: 監査可能なJudge model ID
- `AGENT_EVAL_JUDGE_ALLOWED_HOST`: endpointに許可する単一の完全一致hostname

Judge credentialは `secrets.AGENT_EVAL_JUDGE_API_KEY` だけから取得します。API keyがある場合はHTTPS以外を拒否し、endpoint hostnameが許可hostと完全一致しなければ送信前に停止します。HTTP redirectも拒否するため、許可hostを経由した別hostへの転送は行いません。URL、Bearer token、Source内容はerror messageやreportへ含めません。

## Judge Adapterと出力schema

`StructuredJudgeAdapter` はAgent Runnerと独立した `JudgeTransport` を利用します。標準の外部接続はprovider-neutralな `HttpJudgeTransport`、オフライン試験は `DeterministicMockJudgeTransport` です。Mock Judgeは配線、schema、集計の確認専用であり、そのscoreを品質判断に利用してはいけません。

外部Judgeには、評価に必要な質問、Agent回答、SourceのID・title・URI・snippet、および許可されたTool resultのprojectionが送信されます。したがって、合成データ以外、または機密情報を含み得る評価データを送る場合は、送信先、データ取扱条件、保持期間を確認し、`agent-evaluation` Environmentで承認してから実行してください。

Groundedness Judgeは次の構造を返します。

```json
{
  "schema_version": "1.0",
  "claims": [
    {
      "claim": "セッションタイムアウトは30分である",
      "evaluable": true,
      "supported": true,
      "source_ids": ["def-session"],
      "tool_result_ids": [],
      "reason": "Sourceに同じ設定値がある"
    }
  ]
}
```

`evaluable=false` は `supported=null` とし、unsupported claimと区別します。supported claimは、Judgeへ実際に渡したSource IDまたは成功したTool result IDを最低1つ参照しなければなりません。未知のEvidence ID、schema違反、不正JSONはmalformed responseです。malformed responseだけを最大1回retryし、2回目も不正なら `JudgeMalformedResponseError` としてreportへ残します。unsupportedへ変換しません。

Judge prompt `groundedness.claim-support.v2` は、回答を独立claimへ分解し、供給されたSourceと、policyで許可・projectionされたTool factsだけで支持を判定するよう要求します。Judge requestにはAgentの `confidence`、`critic`、`answer_ok`、自己評価metadataを含めません。source ID、row/source count、Tool名、Tool引数だけではclaimをsupportedにできません。

### Tool Evidence許可ルール

Tool resultはdefault denyです。次の条件をすべて満たす場合だけGroundedness Evidenceへ入ります。

1. Tool名が既知の `structured_query_tool` である。
2. Tool errorがなく、resultがobjectで `success=true` である。
3. `metadata.deterministic is true` である。
4. `metadata.evidence_kind == "structured_query_result"` である。
5. resultに実値を持つ `value`、`aggregate`、`count`、`rows`、`records`、`metrics`など、claimを支持できる許可フィールドがある。
6. projection後のEvidenceが行数、field数、文字列長、payload sizeの上限内である。

Judgeへ送るargumentsは `operation`、`target_metric`、`filters`、`target_dataset`だけです。resultも許可された条件とfact fieldだけへprojectionします。`critic`、`judge`、`confidence`、`answer_ok`、`self_assessment`と同義のネストキー、および `source_ids`、`source_count`などprovenanceだけの値は再帰的に除外します。

`hybrid_search`と`compare_documents`のresultは、明示フラグがあってもTool Evidenceとして使用しません。これらは空でない `SourceTrace.snippet`をEvidence正本とし、snippetを持たないSource ID／titleだけのTraceはGroundedness Evidenceへ送りません。未知Tool、deterministic未設定、source ID／source count／row countしか持たないStructured Queryも除外します。除外されたTool result IDをJudgeが返した場合はunknown evidenceを参照したmalformed responseとして扱い、最大1回のretry後に明示的なJudge errorにします。

各Judge成功時には以下を保存します。

- `judge_model`
- `judge_prompt_version`
- `evaluated_at`
- malformed responseを含む `attempts`

## Groundedness計算式

```text
Groundedness = supported claims / evaluable claims
```

claimが0件、またはevaluable claimが0件の場合は `N/A` です。Judge実行失敗もscore 0にはせず、Judge errorとして分離します。

## Stability計算式

caseごとにRunnerを厳密に `case.repeat` 回呼び、成功Traceから次を比較します。

- Route
- query_type
- Tool nameの呼び出し列
- Tool argument schema／assertionで宣言された主要引数
- Citation IDと解決先Source IDの組の集合
- Answer semantic group

各決定論的dimensionについて、成功runの値を正規化し、次を計算します。

```text
mode share = 最頻値のrun数 / 成功run数
all match = 全repeatが成功 AND distinct value数が1
```

report全体の `all_match_rate` は、`all_match=true` のcase数をStability評価可能case数で割ります。成功runが2件未満ならmode shareは `N/A` です。execution errorが1件でもある場合、残りの成功runが一致していても `all_match=false` とし、errorのrun index、type、messageを別フィールドへ残します。

Answerは文字列一致で比較しません。`stability.semantic-groups.v1` Judge promptへ全回答を渡し、意味的に等価な回答へ同じ `group_id` を割り当てます。そのgroup IDから同じmode shareとall matchを計算します。

合成データでは、direct、definition、structured_query、compare、fallbackから各1ケースを `repeat=3` に設定しています。Fixtureでは同一Traceが返るため、これは実LLMの揺らぎではなくrepeat実行経路の回帰確認です。

## Pricing configとCost

価格表は `config/agent_pricing.json` です。

```json
{
  "schema_version": "1.0",
  "pricing_version": "phase6-synthetic-2026-07-20-v2",
  "currency": "USD",
  "token_unit": 1000000,
  "models": {
    "synthetic-eval-model-v1": {
      "input_usd_per_unit": 1.0,
      "output_usd_per_unit": 2.0
    }
  },
  "tools": {
    "hybrid_search": {"usd_per_call": 0.0}
  }
}
```

```text
input cost  = input tokens  / token_unit × input unit price
output cost = output tokens / token_unit × output unit price
tool cost   = Σ(tool call count × tool unit price)
total cost  = input cost + output cost + tool cost
```

modelは `usage.metadata.model`、Trace metadataの `model` の順で解決します。Agentの `target` 名をmodel名として扱いません。model IDがなければ `model=unreported`、`status=model_not_priced`、推定cost `N/A`です。input/output usageが欠損している場合も、推定costはすべて `N/A` です。未登録modelまたは未登録Toolを0 USDにはせず、statusと `N/A` を保存します。reportには必ず `pricing_version` を含めます。現在の価格表は合成Fixture用であり、実provider価格ではありません。

現行の `ai-agent-rag` Traceはusage metadataへ取得元だけを保存し、実model IDまたはmodel別token usageを保存していません。このため実Agent Costは `model_not_priced`／`N/A`になります。`ai-agent-rag`側でprovider responseから実model IDとmodel別input/output token usageを収集し、`UsageTrace.metadata`へ保存する変更は別Issue候補です。target名へ価格を割り当てる回避策は採用しません。

## 実行方法

APIキーなしのMock Judge実行:

```bash
PYTHONPATH=src python scripts/run_agent_advanced_evaluation.py \
  --runner fixture \
  --judge mock
```

標準出力:

- `.artifacts/agent-advanced/report.json`
- `.artifacts/agent-advanced/report.md`

HTTP Judge実行:

```bash
AGENT_EVAL_JUDGE_API_KEY=... \
PYTHONPATH=src python scripts/run_agent_advanced_evaluation.py \
  --runner fixture \
  --judge http \
  --judge-url https://judge.example.test/evaluate \
  --judge-model judge-model-version \
  --judge-allowed-host judge.example.test
```

HTTP endpointは `JudgeRequest` JSONをPOSTで受け取り、taskに応じたJudge response JSONを直接返します。`--judge-allowed-host`は完全一致で、複数hostが必要なローカル運用ではoptionを繰り返します。API tokenは任意のBearer tokenとして送られ、reportやerror messageには保存しません。

外部integration testは明示的なopt-inです。

```bash
RUN_EXTERNAL_AGENT_JUDGE_TESTS=1 \
AGENT_EVAL_JUDGE_URL=https://judge.example.test/evaluate \
AGENT_EVAL_JUDGE_MODEL=judge-model-version \
AGENT_EVAL_JUDGE_ALLOWED_HOST=judge.example.test \
AGENT_EVAL_JUDGE_API_KEY=... \
PYTHONPATH=src pytest tests/agent_eval/test_external_judge_integration.py -q
```

## 既知の限界

- GroundednessはEvidenceとの意味的支持をJudgeへ委譲するため、Judge model／prompt変更で値が変わります。
- GitHub Environmentのrequired reviewerとVariablesはRepository設定で構成する必要があります。未構成ならAdvanced Jobは安全に接続できません。
- hostname allowlistはapplication層の制御であり、DNS pinning、private network判定、egress firewallの代替ではありません。Runner側でも送信先を制限してください。
- Judge provider側の保存期間、学習利用、監査ログ、削除方針はこのリポジトリでは強制できません。
- 現行 `ai-agent-rag` はStructured Queryの限定fact projectionと実model IDをTraceへ出力しないため、Tool Evidenceはdefault deny、実Agent Costは `N/A` です。
- Mock Judgeの100%は品質を意味しません。
- semantic groupは推移律をJudge出力へ依存します。
- 合成Fixtureのrepeatは実LLMの温度、provider障害、並行Tool実行を再現しません。
- pricing configは手動version管理であり、provider価格を自動取得しません。
- monitor-only指標はPRをblockしません。十分な実データ、Judge再現性、価格表更新手順が確立した後にだけGate昇格を検討します。
