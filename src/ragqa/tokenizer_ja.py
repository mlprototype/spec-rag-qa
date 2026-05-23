from __future__ import annotations

import re

import fugashi
import unidic_lite

# 識別子を形態素分割から保護するパターン
PROTECT_PATTERNS = [
    r"(?<![0-9])[2-5][0-9]{2}(?![0-9])",  # HTTPステータス: 409, 400, 422
    r"(?<![A-Z0-9_])[A-Z][A-Z0-9_]{1,}(?![A-Z0-9_])",  # 大文字識別子
    r"(?<![A-Za-z0-9_])[A-Za-z]+_[A-Za-z0-9_]+(?![A-Za-z0-9_])",  # スネークケース
    r"/[a-zA-Z0-9/]+(?:\{[^}]+\})?",  # APIパス: /api/signup, /users/{id}
    r"(?<![A-Za-z0-9_])[A-Za-z]{2,}(?![A-Za-z0-9_])",  # 英単語
]

# 補足指示書S1: pos1で判定し、固有名詞は名詞に包含させる
ALLOWED_POS = {"名詞", "動詞", "形容詞"}

_PROTECT_REGEXES = [re.compile(p) for p in PROTECT_PATTERNS]


def _build_tagger() -> fugashi.GenericTagger:
    # 補足指示書の方針に従い GenericTagger('') を第一候補にする。
    try:
        return fugashi.GenericTagger("")
    except RuntimeError:
        # 環境にmecabrcが無い場合のフォールバック
        return fugashi.GenericTagger(f'-r /dev/null -d "{unidic_lite.DICDIR}"')


_TAGGER = _build_tagger()


def _protect_with_placeholders(text: str) -> tuple[str, dict[str, str]]:
    protected = text
    placeholder_map: dict[str, str] = {}
    serial = 0

    for i, regex in enumerate(_PROTECT_REGEXES):
        def _repl(match: re.Match[str], pattern_idx: int = i) -> str:
            nonlocal serial
            key = f"__SPEC_{pattern_idx}_{serial}__"
            serial += 1
            placeholder_map[key] = match.group(0)
            # 形態素解析時に分割されないよう、プレースホルダーを単語境界化する
            return f" {key} "

        protected = regex.sub(_repl, protected)

    return protected, placeholder_map


def _feature_pos1(feature: object) -> str | None:
    if hasattr(feature, "pos1"):
        pos1 = getattr(feature, "pos1")
        return str(pos1) if pos1 else None
    if isinstance(feature, (tuple, list)) and len(feature) > 0:
        return str(feature[0]) if feature[0] else None
    return None


def _feature_lemma(feature: object) -> str | None:
    lemma: str | None = None
    if hasattr(feature, "lemma"):
        raw = getattr(feature, "lemma")
        lemma = str(raw) if raw else None
    elif isinstance(feature, (tuple, list)):
        # UniDic系ではlemma相当が index 7 に入るケースが多い
        if len(feature) > 7 and feature[7]:
            lemma = str(feature[7])
        elif len(feature) > 6 and feature[6]:
            lemma = str(feature[6])

    if not lemma or lemma == "*":
        return None
    return lemma


def _consume_placeholder(
    items: list[tuple[str, object]], start: int, placeholder_map: dict[str, str]
) -> tuple[str | None, int]:
    if start + 6 >= len(items):
        return None, start

    s0 = items[start][0]
    s1 = items[start + 1][0]
    s2 = items[start + 2][0]
    s3 = items[start + 3][0]
    s4 = items[start + 4][0]
    s5 = items[start + 5][0]
    s6 = items[start + 6][0]

    if (
        s0 == "__"
        and s1 == "SPEC"
        and s2 == "_"
        and s3.isdigit()
        and s4 == "_"
        and s5.isdigit()
        and s6 == "__"
    ):
        key = f"__SPEC_{s3}_{s5}__"
        if key in placeholder_map:
            return key, start + 7

    return None, start


def tokenize(text: str) -> list[str]:
    """
    日本語BM25向けトークナイズ。
    1) 専門語を保護 2) 形態素解析 3) 品詞フィルタ 4) プレースホルダー復元
    """
    protected, placeholder_map = _protect_with_placeholders(text)

    items = [
        (word.surface, word.feature)
        for word in _TAGGER(protected)
        if word.surface and word.surface.strip()
    ]

    raw_tokens: list[str] = []
    i = 0
    while i < len(items):
        # 補足指示書S3: 解析後トークン列の元位置でインライン復元する
        key, next_i = _consume_placeholder(items, i, placeholder_map)
        if key is not None:
            raw_tokens.append(key)
            i = next_i
            continue

        surface, feature = items[i]
        pos1 = _feature_pos1(feature)
        if pos1 in ALLOWED_POS:
            lemma = _feature_lemma(feature) or surface
            raw_tokens.append(lemma)
        i += 1

    return [placeholder_map.get(token, token) for token in raw_tokens]


def _extract_special_matches(text: str) -> list[str]:
    candidates: list[tuple[int, int, int, str]] = []
    for priority, regex in enumerate(_PROTECT_REGEXES):
        for match in regex.finditer(text):
            candidates.append((priority, match.start(), match.end(), match.group(0)))

    # 優先度高いパターンを先に採用し、重なりは排除
    candidates.sort(key=lambda x: (x[0], x[1], -(x[2] - x[1])))
    selected: list[tuple[int, int, int, str]] = []
    for cand in candidates:
        _, start, end, _ = cand
        overlapped = any(not (end <= s or start >= e) for _, s, e, _ in selected)
        if overlapped:
            continue
        selected.append(cand)

    selected.sort(key=lambda x: x[1])
    ordered_tokens: list[str] = []
    seen: set[str] = set()
    for _, _, _, token in selected:
        if token.isdigit() and len(token) <= 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered_tokens.append(token)
    return ordered_tokens


def detect_special_tokens(query: str) -> list[str]:
    """
    Exact Match Boost対象の識別子トークンを抽出する。
    汎用数字のみ（1桁・2桁）は対象外。
    """
    return _extract_special_matches(query)
