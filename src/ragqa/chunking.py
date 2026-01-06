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
