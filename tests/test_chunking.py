from pathlib import Path

import pytest
from ragqa.chunking import markdown_header_chunks

def test_markdown_header_chunks_preserves_short_final_section():
    text = "# 入力\nemailは必須"
    chunks = markdown_header_chunks("doc1", text, min_chunk_size=50)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_markdown_header_chunks_merges_short_sections_without_content_loss():
    text = "導入\n# 入力\nemail必須\n## 制約\npasswordは8文字以上\n## エラー\n重複時409"
    chunks = markdown_header_chunks("doc1", text, min_chunk_size=50)
    assert chunks
    assert chunks[0].text.startswith("導入\n# 入力\nemail必須\n## 制約")
    indexed_content = "".join("".join(c.text.split()) for c in chunks)
    assert indexed_content == "".join(text.split())
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "doc1" for c in chunks)


def test_markdown_header_chunks_preserves_short_tail_after_long_section():
    text = "# 長い節\n" + "本文。" * 30 + "\n## 必須条件\nUSER_IDを記録する"
    chunks = markdown_header_chunks("doc1", text)
    assert len(chunks) == 2
    assert chunks[1].text == "## 必須条件\nUSER_IDを記録する"
    assert [c.chunk_id for c in chunks] == [0, 1]


@pytest.mark.parametrize("text", ["", "\n\r\n\n", " \t\n \r\n"])
def test_markdown_header_chunks_whitespace_only(text):
    assert markdown_header_chunks("doc1", text) == []


def test_sample_spec_input_conditions_are_indexed():
    path = Path(__file__).resolve().parents[1] / "data/docs/sample_spec.md"
    text = path.read_text(encoding="utf-8")
    chunks = markdown_header_chunks("sample_spec.md", text)
    indexed = "\n".join(c.text for c in chunks)
    assert "## 入力\n- email（必須）\n- password（8文字以上）" in indexed
    assert "".join(indexed.split()) == "".join(text.split())

def test_markdown_header_chunks_codeblock_comment():
    # 2. コードブロック内のコメントによる誤爆
    # コードブロックの中の # コメントがヘッダーとして誤認されて分割されるはず。
    text = """# Header
これはコードです。
```python
# comment
def foo(): pass
```
"""
    chunks = markdown_header_chunks("doc1", text)
    # ヘッダーは1つしかないので、チャンクは1つになるのが正しい
    assert len(chunks) == 1, "Code block comment was mistakenly treated as a Markdown header and split the chunk"

def test_markdown_header_chunks_no_space_and_too_many_hashes():
    # 3. スペース無しの偽ヘッダー
    text = "#見出し\n####### 7つのハッシュ\n本文"
    # いずれも正しいMarkdownヘッダー(h1~h6 + スペース)ではないため、分割されないこと
    chunks = markdown_header_chunks("doc1", text, min_chunk_size=1)
    # 先頭がマッチしないので全体が1つのチャンク（チャンクid=0のみ）か、またはバッファのまま出力される
    assert len(chunks) == 1
    assert "見出し" in chunks[0].text
    assert "7つのハッシュ" in chunks[0].text

def test_markdown_header_chunks_empty_or_newlines():
    # 4. 空ファイルや改行のみのファイル
    assert len(markdown_header_chunks("doc1", "")) == 0
    assert len(markdown_header_chunks("doc1", "\n\r\n\n")) == 0
