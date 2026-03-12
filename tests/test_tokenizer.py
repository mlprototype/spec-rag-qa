import json
import warnings
from pathlib import Path

import fugashi
import unidic_lite

from ragqa.tokenizer_ja import detect_special_tokens, tokenize


def test_tok01_error_code_preserved():
    tokens = tokenize('409エラーが発生した')
    assert '409' in tokens, f'409が保護されていない: {tokens}'


def test_tok02_user_id_preserved():
    tokens = tokenize('USER_IDを確認してください')
    assert 'USER_ID' in tokens, f'USER_IDが保護されていない: {tokens}'


def test_tok03_stopword_removed():
    tokens = tokenize('パスワードは8文字以上')
    assert 'パスワード' in tokens
    assert 'は' not in tokens, f'助詞「は」が残っている: {tokens}'


def test_tok04_reproducibility():
    text = '409エラーが発生した場合はUSER_IDを確認する'
    assert tokenize(text) == tokenize(text), '同一入力で異なる結果'


def test_tok05_detect_special_tokens():
    result = detect_special_tokens('409エラーが発生した')
    assert '409' in result
    result2 = detect_special_tokens('3件ある')
    assert '3' not in result2


def test_tok06_proper_noun_is_kept_as_noun():
    tokens = tokenize('UserとTokyoを確認する')
    assert 'Tokyo' in tokens, f'固有名詞が欠落: {tokens}'


def _get_unidic_version() -> str:
    try:
        tagger = fugashi.GenericTagger('')
    except RuntimeError:
        tagger = fugashi.GenericTagger(f'-r /dev/null -d "{unidic_lite.DICDIR}"')
    info = tagger.dictionary_info
    if not info:
        return 'unknown'
    dic0 = info[0]
    if isinstance(dic0, dict):
        return str(dic0.get('version', 'unknown'))
    return str(getattr(dic0, 'version', 'unknown'))


def test_tokenizer_regression():
    snapshot_path = Path('tests/fixtures/token_snapshot.json')
    if not snapshot_path.exists():
        return

    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    current_ver = _get_unidic_version()

    diffs = []
    cases = snapshot.get('cases', [])
    for case in cases:
        current = tokenize(case['input'])
        expected = case.get('actual_tokens', [])
        if set(current) != set(expected):
            diffs.append(
                {
                    'input': case['input'],
                    'removed': sorted(set(expected) - set(current)),
                    'added': sorted(set(current) - set(expected)),
                }
            )

    if diffs:
        out_path = Path('data/eval/tokenizer_diff_report.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    'version_from': snapshot.get('unidic_version', 'unknown'),
                    'version_to': current_ver,
                    'diffs': diffs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        warnings.warn(f'トークン差分あり: {len(diffs)}件', UserWarning)

        oov_rate = len(diffs) / len(cases) if cases else 0.0
        assert oov_rate <= 0.05, f'OOV率が閾値超過({oov_rate:.0%})。辞書変更を確認してください'
