from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def simple_char_chunks(
    doc_id: str, text: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    text = text.replace("\r\n", "\n")
    chunks: list[Chunk] = []
    i = 0
    cid = 0
    n = len(text)

    while i < n:
        j = min(n, i + chunk_size)
        chunk_text = text[i:j].strip()
        if chunk_text:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk_text))
            cid += 1
        if j == n:
            break
        i = max(0, j - overlap)

    return chunks


def markdown_header_chunks(
    doc_id: str, text: str, min_chunk_size: int = 50
) -> list[Chunk]:
    """
    Markdownのヘッダー(#, ##, ###)単位でテキストを分割する。
    単純な文字数分割よりも、文書の構造（文脈）を保持しやすい。
    """
    # 改行コードの統一
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")

    chunks: list[Chunk] = []
    buffer: list[str] = []
    cid = 0

    # Markdownヘッダーを検知する正規表現
    # ^#{1,6}\s は、「行頭に#が1〜6個あり、その後に空白がある」パターン
    header_pattern = re.compile(r"^#{1,6}\s")

    for line in lines:
        # ヘッダー行を見つけた場合
        if header_pattern.match(line):
            # すでにバッファに中身があり、かつ一定サイズ以上ならチャンクとして保存
            # (min_chunk_sizeは、空の改行やゴミ等の微細なチャンク生成を防ぐため)
            if buffer:
                chunk_text = "\n".join(buffer).strip()
                if len(chunk_text) > 0:
                    chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk_text))
                    cid += 1
                # バッファをリセット
                buffer = []

            # 新しいセクションの開始（ヘッダー行自体もバッファに入れる）
            buffer.append(line)
        else:
            # 普通の行はバッファに追加し続ける
            buffer.append(line)

    # ループ終了後、バッファに残っている最後のセクションを保存
    if buffer:
        chunk_text = "\n".join(buffer).strip()
        if len(chunk_text) > 0:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk_text))

    return chunks
