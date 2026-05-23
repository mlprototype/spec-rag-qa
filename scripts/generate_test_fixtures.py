import json
from pathlib import Path

import fugashi
import unidic_lite

from ragqa.tokenizer_ja import tokenize


CASES = [
    {'input': '409エラーが発生した', 'expected_tokens': ['409']},
    {'input': 'USER_IDを確認してください', 'expected_tokens': ['USER_ID', '確認']},
    {'input': 'パスワードは8文字以上', 'expected_tokens': ['パスワード', '文字']},
]


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


snapshot = {
    'unidic_version': _get_unidic_version(),
    'cases': [{**c, 'actual_tokens': tokenize(c['input'])} for c in CASES],
}

out = Path('tests/fixtures/token_snapshot.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Snapshot saved: {out}')
