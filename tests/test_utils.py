import pytest
from ragqa.utils import _extract_json_object, _parse_fallback

def test_extract_json_object_greedy_match():
    # 1. 欲張りマッチ（Greedy Match）による自爆の誘発
    # 最初に見つかったJSONオブジェクト { "a": 1 } が取得されるのが正しい仕様のはず。
    text = 'ここは { "a": 1 } です。そしてここは { "b": 2 } です。'
    result = _extract_json_object(text)
    assert result == {"a": 1}, "Greedy match likely caught the whole string from the first { to the last }"

def test_extract_json_object_array_input():
    # 2. アレイ（配列）の返却
    # 戻り値の型アノテーションは Dict[str, Any] だが、配列のJSONを渡すとどうなるか。
    text = '[{"verdict": "sufficient"}]'
    result = _extract_json_object(text)
    assert isinstance(result, dict), f"Should return a dictionary, but returned {type(result)}"

def test_parse_fallback_conflict():
    # 3. 矛盾したFallbackと優先度のテスト
    # insufficient と sufficient が両方含まれる場合の挙動確認
    text = "The verdict is not sufficient, it is actually insufficient."
    result = _parse_fallback(text)
    assert result["verdict"] == "insufficient", "Should handle conflicting terms properly or safely default to insufficient"

def test_parse_fallback_edge_cases():
    # 4. 大文字小文字の揺らぎとフォーマット崩れ
    text_camel = '"VerDict" = "SuFfiCient"'
    assert _parse_fallback(text_camel)["verdict"] == "sufficient"

    # フォーマットが崩れているが判定文言が含まれている場合
    text_broken = "```json\n { broken "
    # テキストフォールバックに渡されるので safe default の insufficient になるはず
    assert _parse_fallback(text_broken)["verdict"] == "insufficient"
