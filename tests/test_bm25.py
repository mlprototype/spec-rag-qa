import pytest

from ragqa.bm25_store import BM25Store
from ragqa.chunking import Chunk


@pytest.fixture
def store_with_corpus():
    chunks = [
        Chunk('test.md', 0, 'パスワードは8文字以上必須'),
        Chunk('test.md', 1, '409エラーはメール重複を示す'),
        Chunk('test.md', 2, 'セッションタイムアウトは30分'),
        Chunk('test.md', 3, '409エラーが継続する場合、409エラーのトリアージを行う'),
        Chunk('test.md', 4, '監査ログにはUSER_IDを記録する'),
    ]
    store = BM25Store(b=0.75, k1=2.0)
    store.build(chunks)
    return store


def test_bm25_01_reproducibility(store_with_corpus):
    s1 = store_with_corpus.score(1, ['409'])
    s2 = store_with_corpus.score(1, ['409'])
    assert abs(s1 - s2) < 1e-9


def test_bm25_02_relevant_higher(store_with_corpus):
    hit = store_with_corpus.score(1, ['409'])
    miss = store_with_corpus.score(2, ['409'])
    assert hit > miss


def test_bm25_03_higher_tf_higher_score(store_with_corpus):
    once = store_with_corpus.score(1, ['409'])
    many = store_with_corpus.score(3, ['409'])
    assert many > once


def test_bm25_04_length_normalization():
    chunks = [
        Chunk('t.md', 0, '409エラー'),
        Chunk('t.md', 1, '409エラーが発生した場合はUSER_IDを確認しログを調査しサポートに連絡する'),
        Chunk('t.md', 2, 'セッション管理の運用手順'),
        Chunk('t.md', 3, '監査ログ保持期間は90日'),
        Chunk('t.md', 4, '会員登録仕様の改訂履歴'),
    ]
    store = BM25Store(b=0.75, k1=2.0)
    store.build(chunks)

    score_short = store.score(0, ['409'])
    score_long = store.score(1, ['409'])
    assert score_short > score_long, (
        f'短チャンクが高スコアになるはず: short={score_short:.4f}, long={score_long:.4f}'
    )


def test_bm25_05_k1_affects_score():
    chunks = [
        Chunk('t.md', 0, '409エラー 409エラー 409エラー'),
        Chunk('t.md', 1, 'セッション管理'),
        Chunk('t.md', 2, 'パスワード強度'),
        Chunk('t.md', 3, '運用監査ログ'),
        Chunk('t.md', 4, '通知設定'),
    ]

    store_low = BM25Store(b=0.75, k1=1.0)
    store_high = BM25Store(b=0.75, k1=3.0)
    store_low.build(chunks)
    store_high.build(chunks)

    score_low = store_low.score(0, ['409'])
    score_high = store_high.score(0, ['409'])
    assert score_low != score_high, (
        f'k1変更でスコアが変化しない場合は実装バグ: k1=1.0→{score_low:.4f}, k1=3.0→{score_high:.4f}'
    )


def test_bm25_06_oov_no_error(store_with_corpus):
    score = store_with_corpus.score(0, ['存在しないトークンXYZ'])
    assert score == 0.0


def test_bm25_07_empty_query(store_with_corpus):
    assert store_with_corpus.score(0, []) == 0.0


def test_bm25_08_exact_match_boost():
    chunks = [Chunk('t.md', 0, '409エラー'), Chunk('t.md', 1, 'セッション')]
    store = BM25Store()
    store.build(chunks)

    raw = store.search('409', top_k=2, boost_alpha=0.0, boost_beta=0.0)[0]['bm25_score']
    boosted = store.search('409', top_k=2, boost_alpha=1.0, boost_beta=0.0)[0]['bm25_score']
    assert boosted > raw, f'Boostが効いていない: raw={raw}, boosted={boosted}'
