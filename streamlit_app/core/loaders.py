from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import pdfplumber

from .config import AppPaths
from .text_clean import post_clean_text


@dataclass
class Chunk:
    doc_id: str
    page: int
    chunk_id: str
    text: str
    section_path: Optional[str] = None


def _norm_doc_id(s: str) -> str:
    s = str(s).strip().strip('"').strip()
    s = re.sub(r"\s+", " ", s)
    return s

_PAGE_HINT_RE = re.compile(r"\(p\.\s*(\d{1,4})\)")

def _infer_page_from_text(text: str) -> int:
    """
    일부 전처리(pp_v5) 산출물은 page 메타가 1로 고정되는 케이스가 있어,
    텍스트 내 '(p.xx)' 힌트에서 대표 페이지를 추정한다.
    """
    if not text:
        return 0
    nums = [int(x) for x in _PAGE_HINT_RE.findall(text)]
    if not nums:
        return 0
    # 가장 자주 나온 페이지를 사용하고, 동률이면 앞에서 먼저 등장한 값을 사용
    counts: Dict[int, int] = {}
    for n in nums:
        counts[n] = counts.get(n, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], nums.index(kv[0])))[0][0]
    return int(best)


def make_pdf_map(pdf_dir: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in sorted(pdf_dir.glob("*.pdf")):
        m[_norm_doc_id(p.name)] = p
    return m


def load_chunks_from_jsonl(jsonl_path: Path, doc_id: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            page = obj.get("page", obj.get("page_label", 0))
            try:
                page = int(float(page))
            except Exception:
                page = 0

            text = obj.get("text", obj.get("content", "")) or ""
            text = post_clean_text(str(text))
            hinted_page = _infer_page_from_text(text)
            if page <= 0 and hinted_page > 0:
                page = hinted_page
            elif page == 1 and hinted_page >= 2:
                # page=1 고정 노이즈를 힌트 페이지로 보정
                page = hinted_page

            if page <= 0 or not text:
                continue

            chunk_id = str(obj.get("chunk_id", f"{Path(doc_id).stem}__p{page:04d}_c{i:04d}"))
            section_path = obj.get("section_path") or obj.get("section") or None

            chunks.append(Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=chunk_id,
                text=text,
                section_path=section_path,
            ))
    return chunks


def extract_page_texts_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = post_clean_text(txt)
            if txt:
                out.append((pidx, txt))
    return out


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=함\.)\s+|(?<=됨\.)\s+|(?<=음\.)\s+"
)

def _split_sentences(text: str) -> List[str]:
    t = post_clean_text(text or "")
    if not t:
        return []
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(t) if s and s.strip()]
    return sents

def _tail_overlap_text(prev_chunk: str, overlap_chars: int) -> str:
    if not prev_chunk or overlap_chars <= 0:
        return ""
    if len(prev_chunk) <= overlap_chars:
        return prev_chunk
    start = max(0, len(prev_chunk) - overlap_chars)
    ws = prev_chunk.find(" ", start)
    if ws != -1 and ws + 1 < len(prev_chunk):
        start = ws + 1
    return prev_chunk[start:].strip()

def _chunk_text_sentence_with_overlap(text: str, chunk_length: int, overlap_chars: int) -> List[str]:
    sents = _split_sentences(text)
    if not sents:
        return []

    base_chunks: List[str] = []
    cur = ""
    for sent in sents:
        if len(sent) > chunk_length:
            if cur:
                base_chunks.append(cur.strip())
                cur = ""
            for i in range(0, len(sent), chunk_length):
                piece = sent[i:i + chunk_length].strip()
                if piece:
                    base_chunks.append(piece)
            continue

        if not cur:
            cur = sent
            continue

        cand = f"{cur} {sent}"
        if len(cand) <= chunk_length:
            cur = cand
        else:
            base_chunks.append(cur.strip())
            cur = sent

    if cur:
        base_chunks.append(cur.strip())

    if len(base_chunks) <= 1:
        return base_chunks

    out: List[str] = [base_chunks[0]]
    for i in range(1, len(base_chunks)):
        tail = _tail_overlap_text(base_chunks[i - 1], overlap_chars)
        merged = f"{tail} {base_chunks[i]}".strip() if tail else base_chunks[i]
        out.append(merged)
    return out


def _page_texts_to_runtime_c1_chunks(doc_id: str, page_texts: List[Tuple[int, str]], chunk_length: int) -> List[Chunk]:
    """
    ✅ runtime C1:
    - 페이지를 유지한 상태로 문장 경계 기준 청킹 + overlap 적용
    - 서비스의 페이지 하이라이트/근거 보기 UX를 유지하기 위함
    """
    out: List[Chunk] = []
    doc_id_n = _norm_doc_id(doc_id)
    chunk_length = max(50, int(chunk_length or 1200))
    overlap_chars = max(80, min(150, chunk_length // 6))

    for page, txt in page_texts:
        t = post_clean_text(txt)
        if not t:
            continue
        pieces = _chunk_text_sentence_with_overlap(t, chunk_length=chunk_length, overlap_chars=overlap_chars)
        for ci, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunk_id = f"{Path(doc_id_n).stem}__p{int(page):04d}_c{ci:04d}__rt{chunk_length}"
            out.append(Chunk(
                doc_id=doc_id_n,
                page=int(page),
                chunk_id=chunk_id,
                text=piece,
                section_path=None,
            ))
    return out


def get_chunks(
    doc_id: str,
    pdf_path: Optional[str],
    paths: AppPaths,
    source: str,
    *,
    precomputed_jsonl: Optional[Path] = None,
    chunk_length: Optional[int] = None,  # ✅ runtime_c1에 사용
) -> Tuple[List[Chunk], Path]:
    """
    source:
      - precomputed_chunks : precomputed_jsonl 필요
      - pdf_fallback       : 페이지 단위 chunk
      - runtime_c1         : 페이지 유지 + chunk_length 기준 문자청킹
    Returns: (chunks, artifact_path)
    """
    doc_id_n = _norm_doc_id(doc_id)

    if source == "precomputed_chunks":
        if precomputed_jsonl is None:
            raise ValueError("precomputed_chunks source에는 precomputed_jsonl 경로를 넘겨야 합니다.")
        if not precomputed_jsonl.exists():
            raise FileNotFoundError(f"precomputed jsonl not found: {precomputed_jsonl}")
        chunks = load_chunks_from_jsonl(precomputed_jsonl, doc_id_n)
        return chunks, precomputed_jsonl

    if source == "pdf_fallback":
        if not pdf_path:
            raise FileNotFoundError("pdf_path가 없어서 pdf_fallback으로 로드할 수 없습니다.")
        page_texts = extract_page_texts_from_pdf(pdf_path)
        chunks = [
            Chunk(doc_id=doc_id_n, page=p, chunk_id=f"{Path(doc_id_n).stem}__p{p:04d}_pagechunk", text=t)
            for p, t in page_texts
        ]
        return chunks, Path(pdf_path)

    if source == "runtime_c1":
        if not pdf_path:
            raise FileNotFoundError("pdf_path가 없어서 runtime_c1으로 로드할 수 없습니다.")
        page_texts = extract_page_texts_from_pdf(pdf_path)
        chunks = _page_texts_to_runtime_c1_chunks(doc_id_n, page_texts, int(chunk_length or 1200))
        return chunks, Path(pdf_path)

    raise ValueError(f"Unknown source: {source}")
