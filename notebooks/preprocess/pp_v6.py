"""
v1: format_table, 단순 메타데이터, 텍스트 800자 청크
v2: format_table과 단순 메타데이터, 섹션 기반 청크
v3: 메타데이터 텍스트 정제 작업 추가
v4: 이후 추가
- 표 좀더 다듬어보기
- 목차 내용 없애버리기(O)
- 텍스트 정제 고도화
- 타이틀(프로젝트 이름) 완벽하게 가져오기
- 이미지 추출
v5: pdf 파서 교체(-> docling)
- 의미 기반 청킹 (section_path 단위)
- 표 복원 (table_row → 하나의 블록)
- chunk() 문장 경계 분할
- 함수 정리 (중복 제거)
v6: 표 위치 수정
"""

from preprocess.pp_basic import BASE_DIR, RAW_DIR, docs

import re
import os
import json
from pathlib import Path
import pdfplumber
import faiss


pp_version = "5"


# ── 데이터 로딩 (1회) ──

all_data_path = os.path.join(BASE_DIR, "data", f"ALL_DATA_v{pp_version}.json")

with open(all_data_path, "r", encoding="utf-8") as f:
    _raw = json.load(f)

if isinstance(_raw, list):
    ALL_DATA = {}
    for item in _raw:
        doc = item.get("metadata", {}).get("document", "")
        if doc:
            ALL_DATA.setdefault(doc, []).append(item)
else:
    ALL_DATA = _raw


# ── 내부 함수 ──

def _clean_text(text: str) -> str:
    """PDF 추출 텍스트에서 노이즈 제거."""
    text = re.sub(r'^- \d+ -\n?', '', text.strip())
    text = re.sub(r'((\S)\s+){3,}\2', '', text)
    text = re.sub(r'(.)\1{4,}', '', text)
    text = re.sub(r'[·.…]{5,}', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _fix_line_break_splits(text: str) -> str:
    """PDF 줄바꿈으로 인한 단어 분리 복원."""
    # 1. 한글-줄바꿈-한글: 이어붙임
    text = re.sub(r'(?<=[\uAC00-\uD7A3])\n(?=[\uAC00-\uD7A3])', '', text)
    
    # 2. 한글-공백+줄바꿈-한글: 이어붙임 (trailing space 케이스)
    text = re.sub(r'(?<=[\uAC00-\uD7A3]) *\n(?=[\uAC00-\uD7A3])', '', text)
    
    return text


def _extract_text(pdf_path: Path | str) -> str:
    """pdfplumber로 PDF 텍스트 추출. ALL_DATA에 없는 문서용 fallback."""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _chunk_by_sentence(text: str, size: int = 800) -> list[str]:
    """문장 경계 기반 청킹. ALL_DATA에 없는 문서용 fallback."""
    sentences = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s|(?<=함\.)\s|(?<=음\.)\s|(?<=됨\.)\s', text)

    chunks = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(buffer) + len(sent) + 1 <= size:
            buffer = f"{buffer} {sent}" if buffer else sent
        else:
            if buffer:
                chunks.append(buffer.strip())
            if len(sent) > size:
                for i in range(0, len(sent), size):
                    chunks.append(sent[i:i + size].strip())
                buffer = ""
            else:
                buffer = sent

    if buffer:
        chunks.append(buffer.strip())

    return chunks


def _format_table(table: dict) -> str:
    """table_content를 읽기 좋은 텍스트로 변환. _chunk_legacy 전용."""
    tc = table.get("table_content", {})
    cols = tc.get("columns", [])
    rows = tc.get("data", [])
    if not cols and not rows:
        return ""

    lines = []
    title = table.get("table_title", "")
    if title:
        lines.append(f"[표] {title}")
    lines.append(" | ".join(str(c or "") for c in cols))
    for row in rows:
        lines.append(" | ".join(str(c or "") for c in row))
    return "\n".join(lines)


def _chunk_legacy(data: dict, size: int = 800) -> list[str]:
    """기존 포맷 청킹 (v4 이하 ALL_DATA 호환)."""
    pages = data["metadata"]

    sections = []
    current_sec = None
    current_parts = []

    for page in pages:
        sec = page.get("section")
        if isinstance(sec, list):
            sec_str = " | ".join(sec)
        elif sec:
            sec_str = sec
        else:
            sec_str = ""

        if sec_str != current_sec and current_parts:
            sections.append((current_sec, "\n".join(current_parts)))
            current_parts = []
        current_sec = sec_str

        parts = []
        if sec_str:
            parts.append(f"[섹션: {sec_str}] (p.{page['page'] + 1})")
        else:
            parts.append(f"(p.{page['page'] + 1})")

        if page.get("text"):
            parts.append(_clean_text(_fix_line_break_splits(page["text"])))

        if page.get("table"):
            for t in page["table"]:
                table_text = _format_table(t)
                if table_text:
                    parts.append(table_text)

        current_parts.append("\n".join(parts))

    if current_parts:
        sections.append((current_sec, "\n".join(current_parts)))

    chunks = []
    for sec_name, sec_text in sections:
        for i in range(0, len(sec_text), size):
            chunks.append(sec_text[i:i + size])

    return chunks


_TABLE_TITLE_PATTERN = re.compile(r"(별표\s*\d+|표\s*\d+(?:[.\-]\d+)*)")


def _find_table_title(items: list[dict], anchor_idx: int, page: int) -> str:
    """table_row 주변 caption에서 표 제목을 추정."""
    def _is_table_like_caption(text: str) -> bool:
        t = text.strip()
        return bool(
            t and (
                _TABLE_TITLE_PATTERN.search(t)
                or "총괄표" in t
                or "목록표" in t
                or "표준" in t and "표" in t
            )
        )

    # 우선순위: 같은 페이지에서 직전 caption 탐색
    for i in range(anchor_idx - 1, max(-1, anchor_idx - 40), -1):
        meta = items[i].get("metadata", {})
        if abs((meta.get("page", -999) or 0) - page) > 1:
            continue
        if meta.get("type") == "table_row":
            continue
        cand = _clean_text(items[i].get("content", "") or "")
        if _is_table_like_caption(cand):
            return cand

    # 보조: 같은 페이지에서 직후 caption 탐색
    for i in range(anchor_idx + 1, min(len(items), anchor_idx + 15)):
        meta = items[i].get("metadata", {})
        if abs((meta.get("page", -999) or 0) - page) > 1:
            continue
        if meta.get("type") == "table_row":
            continue
        cand = _clean_text(items[i].get("content", "") or "")
        if _is_table_like_caption(cand):
            return cand

    return ""


def _chunk_docling(items: list[dict], size: int = 800) -> list[str]:
    """Docling 포맷 의미 기반 청킹: section_path + 표 복원."""

    # 1) 표 복원: table_id별로 행을 합침
    tables = {}
    for idx, item in enumerate(items):
        meta = item.get("metadata", {})
        if meta.get("type") == "table_row" and meta.get("table_id") is not None:
            tables.setdefault(meta["table_id"], []).append((idx, item))

    table_blocks = {}
    for tid, rows in tables.items():
        rows.sort(key=lambda x: x[1].get("metadata", {}).get("row", 0))
        row_items = [r[1] for r in rows]
        first_idx = rows[0][0]
        first_meta = row_items[0].get("metadata", {})
        page = first_meta.get("page", 0)
        title = _find_table_title(items, first_idx, page)
        table_head = f"[표] {title}".strip() if title else f"[표] p.{page} table_{tid}"
        combined = "\n".join(r.get("content", "") for r in row_items)
        table_blocks[tid] = {
            "content": f"{table_head}\n{combined}",
            "metadata": first_meta,
            "_anchor_idx": first_idx,
        }

    # 2) table_row → table_block 치환 후 page 기준으로 재정렬
    #    (Docling 산출에서 table_row가 문서 후반에 몰리는 케이스 보정)
    merged_entries = []
    seen_tables = set()
    for idx, item in enumerate(items):
        meta = item.get("metadata", {})
        if meta.get("type") == "table_row":
            tid = meta.get("table_id")
            if tid not in seen_tables and tid in table_blocks:
                seen_tables.add(tid)
                block = table_blocks[tid]
                bmeta = block.get("metadata", {})
                merged_entries.append((
                    int(bmeta.get("page", 10**9) or 10**9),
                    1,  # 같은 페이지에서는 본문 뒤에 표 배치
                    int(block.get("_anchor_idx", idx)),
                    {"content": block["content"], "metadata": bmeta},
                ))
            continue

        merged_entries.append((
            int(meta.get("page", 10**9) or 10**9),
            0,
            idx,
            item,
        ))

    merged_entries.sort(key=lambda x: (x[0], x[1], x[2]))
    all_items = [entry[3] for entry in merged_entries]

    # 3) 의미 단위(segment) 구성: section_path가 바뀌면 새 세그먼트
    #    표는 독립 세그먼트로 분리해서 위치/형태를 보존
    segments = []
    current_path = None
    current_parts = []

    for item in all_items:
        content = item.get("content", "").strip()
        if not content:
            continue

        meta = item.get("metadata", {})
        label = meta.get("label", "")
        path = meta.get("section_path", "") or ""
        page = meta.get("page", 0)

        # 노이즈 필터링
        if len(content) <= 3 and label != "section_header":
            continue

        is_table_block = content.startswith("[표]")

        # section_path 전환 → 새 세그먼트
        if path != current_path and current_parts:
            segments.append("\n".join(current_parts))
            current_parts = []
        current_path = path

        if is_table_block:
            # 표는 본문과 합치지 않고 독립 세그먼트로 유지
            if current_parts:
                segments.append("\n".join(current_parts))
                current_parts = []
            segments.append(content)
            continue

        # section_header는 제목으로 표시
        if label == "section_header":
            current_parts.append(f"\n## {content} (p.{page})")
        else:
            current_parts.append(_clean_text(_fix_line_break_splits(content)))

    if current_parts:
        segments.append("\n".join(current_parts))


    # 4) 세그먼트 → 청크 
    chunks = []
    buffer = ""

    # # 4.1) 작은 건 합치고, 큰 건 분할
    # for seg in segments:
    #     if len(buffer) + len(seg) + 1 <= size:
    #         buffer = f"{buffer}\n{seg}" if buffer else seg
    #     else:
    #         if buffer:
    #             chunks.append(buffer.strip())
    #         if len(seg) > size:
    #             for i in range(0, len(seg), size):
    #                 chunks.append(seg[i:i + size].strip())
    #             buffer = ""
    #         else:
    #             buffer = seg


    # # 4.2) 작은 건 합치되, 큰 세그먼트는 통째로 유지
    # for seg in segments:
    #     if len(buffer) + len(seg) + 1 <= size:
    #         buffer = f"{buffer}\n{seg}" if buffer else seg
    #     else:
    #         if buffer:
    #             chunks.append(buffer.strip())
    #         # 큰 세그먼트도 그냥 통째로 넣기
    #         chunks.append(seg.strip())
    #         buffer = ""


    # 4.3) 표는 항상 독립 청크, 텍스트만 문장 경계 분할
    for seg in segments:
        if seg.startswith("[표]"):
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.append(seg.strip())
            continue

        if len(buffer) + len(seg) + 1 <= size:
            buffer = f"{buffer}\n{seg}" if buffer else seg
        else:
            if buffer:
                chunks.append(buffer.strip())

            if len(seg) <= size * 2:
                # 표이거나 size의 2배 이내면 통째로
                chunks.append(seg.strip())
                buffer = ""
            else:
                # 긴 텍스트만 문장 단위 분할
                sub_chunks = _chunk_by_sentence(seg, size)
                chunks.extend(sub_chunks)
                buffer = ""


    if buffer:
        chunks.append(buffer.strip())

    return chunks


def _chunk_from_alldata(doc_name: str, all_data: dict, size: int = 800) -> list[str] | None:
    """ALL_DATA에서 문서별 청크 생성. Docling/기존 포맷 자동 감지."""
    if doc_name not in all_data:
        print(f"찾는 문서 없음: {doc_name}")
        return None

    data = all_data[doc_name]

    if isinstance(data, list) and len(data) > 0 and "content" in data[0]:
        return _chunk_docling(data, size)

    if isinstance(data, dict) and "metadata" in data:
        return _chunk_legacy(data, size)

    return None


def _build_index(chunks: list[str], embed_model):
    """FAISS 인덱스 생성."""
    embs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs.astype("float32"))
    return index, chunks


# ── 공개 API (notebook에서 호출) ──

def gen_input(doc_path, embed_model):
    """문서 1개 → (FAISS index, chunks). notebook의 유일한 진입점."""
    chunks = _chunk_from_alldata(doc_path.name, ALL_DATA)

    if chunks is None:
        text = _fix_line_break_splits(_extract_text(doc_path))
        text = _clean_text(text)
        chunks = _chunk_by_sentence(text)

    index, chunks = _build_index(chunks, embed_model)
    return index, chunks


def show_sample(docs, n=5, show_table_summary=True):
    """디버깅용: 첫 문서의 청크 샘플 출력."""
    test_name = docs[0].name
    chunks_test = _chunk_from_alldata(test_name, ALL_DATA)

    if chunks_test is None:
        text = _fix_line_break_splits(_extract_text(docs[0]))
        text = _clean_text(text)
        chunks_test = _chunk_by_sentence(text)

    print(f"문서: {test_name}")
    print(f"총 청크 수: {len(chunks_test)}")
    if show_table_summary:
        table_idxs = [i for i, c in enumerate(chunks_test) if "[표]" in c]
        print(f"표 포함 청크 수: {len(table_idxs)}")
        if table_idxs:
            preview = table_idxs[:10]
            suffix = " ..." if len(table_idxs) > 10 else ""
            print(f"표 청크 인덱스: {preview}{suffix}")

    if n == "all":
        n = len(chunks_test)

    for i in range(min(n, len(chunks_test))):
        print(f"\n=== 청크 {i} ({len(chunks_test[i])}자) ===")
        print(chunks_test[i])


__all__ = [
    "ALL_DATA",
    "clean_text",
    "fix_line_break_splits",
    "extract_text",
    "chunk_from_alldata",
    "chunk_docling",
]

def clean_text(text: str) -> str:
    return _clean_text(text)

def fix_line_break_splits(text: str) -> str:
    return _fix_line_break_splits(text)

def extract_text(pdf_path: Path | str) -> str:
    return _extract_text(pdf_path)

def chunk_from_alldata(doc_name: str, size: int = 800) -> list[str] | None:
    return _chunk_from_alldata(doc_name, ALL_DATA, size)

def chunk_docling(items: list[dict], size: int = 800) -> list[str]:
    return _chunk_docling(items, size)
