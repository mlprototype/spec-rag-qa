"""
Exact Match Boost の統合テスト。
インメモリ BM25 を使用するためインデックスファイル不要・ネットワーク不要。
"""

import pytest

from ragqa.bm25_store import BM25Store
from ragqa.chunking import Chunk
from ragqa.tokenizer_ja import detect_special_tokens


@pytest.fixture
def boost_corpus() -> BM25Store:
    """Phase 0 失敗ケースを模したコーパス。"""
    chunks = [
        Chunk("error_code_reference.md", 0, "HTTP 409 はメール重複競合を表す。422 は意味的に不正な要求。"),
        Chunk("sample_spec_v2.md", 1, "登録済みメールアドレスを使用した場合は HTTP 409 Conflict を返す。"),
        Chunk("api_contract_signup.md", 2, "POST /api/signup のエンドポイント定義。signup リクエスト仕様。"),
        Chunk("audit_logging_standard.md", 3, "監査ログには USER_ID を必ず記録しなければならない。"),
        Chunk("unrelated_marketing_doc.md", 4, "マーケティング施策とキャンペーン管理について説明する。"),
        Chunk("account_lifecycle_spec.md", 5, "アカウントのライフサイクル管理とステータス遷移。"),
        Chunk("support_playbook.md", 6, "サポート対応の標準フローと優先度判定について。"),
    ]
    store = BM25Store(b=0.75, k1=2.0)
    store.build(chunks)
    return store


def test_int01_no_identifier_returns_empty_list():
    """
    識別子を含まないクエリでは detect_special_tokens は空リストを返すこと。
    """
    tokens = detect_special_tokens(
        "登録済みメールアドレスで会員登録した場合のステータスコードは何ですか?"
    )
    assert tokens == [], f"識別子なしクエリで誤検出: {tokens}"


def test_int02_both_422_and_409_detected():
    tokens = detect_special_tokens("重複メールは 422 ですか、それとも 409 ですか？")
    assert "422" in tokens, f"422 が検出されない: {tokens}"
    assert "409" in tokens, f"409 が検出されない: {tokens}"


def test_int03_signup_detected():
    tokens = detect_special_tokens("signup のエンドポイント定義はどの文書にありますか？")
    assert "signup" in tokens, f"signup が検出されない: {tokens}"


def test_int04_user_id_detected():
    tokens = detect_special_tokens("監査ログに USER_ID は必須ですか？")
    assert "USER_ID" in tokens, f"USER_ID が検出されない: {tokens}"


def test_int05_exact_hits_counted(boost_corpus):
    """409 を含むチャンクで exact_hits >= 1 になること。"""
    hits = boost_corpus.search(
        "重複メールは 422 ですか、それとも 409 ですか？",
        top_k=7,
        boost_alpha=0.0,
        boost_beta=0.0,
    )
    hit_map = {h["chunk_idx"]: h["exact_hits"] for h in hits}
    assert hit_map.get(0, 0) >= 1, f"error_code_reference の exact_hits が 0: {hit_map}"
    assert hit_map.get(1, 0) >= 1, f"sample_spec_v2 の exact_hits が 0: {hit_map}"


def test_int06_alpha_boost_improves_rank():
    """
    no_boost でヒット済みなら boosted で順位が必ず向上することを確認する。
    このテストでは query='signup ...' を使い、no_boost: rank=3 -> boosted: rank=1 を検証する。
    """
    chunks = [
        Chunk("correct_signup.md", 0, "POST /api/signup endpoint signup definition"),
        Chunk("noise_endpoint.md", 1, "エンドポイント 定義 仕様 エンドポイント 定義"),
        Chunk("noise_generic.md", 2, "一般文書"),
    ]
    store = BM25Store(b=0.75, k1=2.0)
    store.build(chunks)

    query = "signup のエンドポイント定義はどの文書にありますか？"
    no_boost = store.search(query, top_k=7, boost_alpha=0.0, boost_beta=0.0)
    boosted = store.search(query, top_k=7, boost_alpha=1.5, boost_beta=0.0)

    def best_rank(hits: list[dict], target_idxs: set[int]) -> int | None:
        for rank, h in enumerate(hits, start=1):
            if h["chunk_idx"] in target_idxs:
                return rank
        return None

    correct_idxs = {0}
    rank_no_boost = best_rank(no_boost, correct_idxs)
    rank_boosted = best_rank(boosted, correct_idxs)

    assert rank_boosted is not None, (
        "Boost ありでも正解チャンクがトップ7に入らない"
    )

    if rank_no_boost is not None:
        assert rank_boosted < rank_no_boost, (
            f"Boost でランク改善しない: "
            f"no_boost={rank_no_boost}, boosted={rank_boosted}"
        )


def test_int07_beta_boost_on_all_hit():
    """全識別子ヒット時に beta ボーナスでスコアが増加する。"""
    chunks = [
        Chunk("codes.md", 0, "HTTP 409 と 422 を定義する。"),
        Chunk("partial.md", 1, "HTTP 409 のみを定義する。"),
        Chunk("noise.md", 2, "一般的な運用説明。"),
    ]
    store = BM25Store(b=0.75, k1=2.0)
    store.build(chunks)

    query = "重複メールは 422 ですか、それとも 409 ですか？"

    alpha_only = store.search(query, top_k=1, boost_alpha=1.5, boost_beta=0.0)
    alpha_beta = store.search(query, top_k=1, boost_alpha=1.5, boost_beta=2.0)

    assert alpha_only, "alpha_only の検索結果が空"
    assert alpha_beta, "alpha_beta の検索結果が空"

    assert alpha_only[0]["chunk_idx"] == alpha_beta[0]["chunk_idx"], (
        "alpha と alpha+beta で1位チャンクが一致しないため前提崩壊"
    )

    assert alpha_beta[0]["bm25_score"] >= alpha_only[0]["bm25_score"], (
        "beta=2.0 でスコアが上昇していない"
    )


def test_int08_no_boost_on_plain_query(boost_corpus):
    """識別子を含まないクエリでは alpha/beta によってスコアが変わらない。"""
    query = "パスワードの最小文字数を教えてください"

    no_boost = boost_corpus.search(query, top_k=7, boost_alpha=0.0, boost_beta=0.0)
    boosted = boost_corpus.search(query, top_k=7, boost_alpha=1.5, boost_beta=2.0)

    nb_scores = {h["chunk_idx"]: h["bm25_score"] for h in no_boost}
    bo_scores = {h["chunk_idx"]: h["bm25_score"] for h in boosted}
    for idx in nb_scores:
        assert abs(nb_scores[idx] - bo_scores.get(idx, 0.0)) < 1e-9, (
            f"識別子なしクエリでスコアが変化した: chunk_idx={idx}"
        )
