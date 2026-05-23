import pytest
from ragqa.chunking import markdown_header_chunks, Chunk

def test_markdown_header_chunks_unused_min_size():
    # 1. 使われていない引数の露呈 (Dead Code)
    # min_chunk_size=50 だが、中身で適用されていないため小さなチャンクが生成されてしまうはず。
    text = "# A\nB"
    chunks = markdown_header_chunks("doc1", text, min_chunk_size=50)
    for chunk in chunks:
        assert len(chunk.text) >= 50, f"Chunk size {len(chunk.text)} is less than min_chunk_size(50). The argument is ignored!"

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
