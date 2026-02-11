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
"""

from preprocess.pp_basic import BASE_DIR, RAW_DIR, docs

import re
import os
import json
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss


pp_version = "5"


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
    ALL_DATA = _raw  # 기존 포맷 그대로

# index_pages_path = os.path.join(BASE_DIR, "data", "01_index_pages.json")

# with open(index_pages_path, "r", encoding="utf-8") as f:
#     index_pages = json.load(f)



# 기본 함수
# Chunking
def chunk(text: str, size: int = 800) -> list[str]:
    """길이 기반 청킹. 문장 경계에서 끊어서 size 이내로 분할."""
    sentences = re.split(r'(?<=[.다함음임됨짐림nim]\.?\s)', text)
    
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
            # 문장 자체가 size보다 크면 강제 분할
            if len(sent) > size:
                for i in range(0, len(sent), size):
                    chunks.append(sent[i:i + size].strip())
                buffer = ""
            else:
                buffer = sent
    
    if buffer:
        chunks.append(buffer.strip())
    
    return chunks


# PDF to text
def extract_text(pdf_path: Path | str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)



def build_index(chunks: list[str], embed_model):
    embs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs.astype("float32"))
    return index, chunks


def gen_input(doc_path, embed_model):
    chunks = chunk_from_alldata(doc_path.name, ALL_DATA)

    if chunks is None:
        text = clean_text(extract_text(doc_path))
        chunks = chunk(text)

    index, chunks = build_index(chunks, embed_model)

    return index, chunks



def gen_doc_indexes(docs, embed_model):

    doc_indexes = {}

    for doc_path in docs:
        print(f"처리 중: {doc_path.name}")

        chunks = chunk_from_alldata(doc_path.name, ALL_DATA)

        if chunks is None:
            text = clean_text(extract_text(doc_path))
            chunks = chunk(text)

        index, chunks_list = build_index(chunks, embed_model)
        doc_indexes[doc_path] = (index, chunks_list)

    print("모든 문서 인덱싱 완료")

    return doc_indexes


def format_table(table: dict) -> str:
    """table_content를 읽기 좋은 텍스트로 변환."""
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


def clean_text(text: str) -> str:
    """PDF 추출 텍스트에서 노이즈 제거."""
    # 페이지 번호 (- 1 -, - 23 - 등)
    text = re.sub(r'^- \d+ -\n?', '', text.strip())
    # 같은 글자가 공백 끼고 반복 (목   목   목...)
    text = re.sub(r'((\S)\s+){3,}\2', '', text)
    # 같은 글자 연속 5회+ (차차차차차...)
    text = re.sub(r'(.)\1{4,}', '', text)
    # 목차 점선 (···, ……, .... 등)
    text = re.sub(r'[·.…]{5,}', ' ', text)
    # 같은 글자 5회 이상 반복 (목   목   목... / 차차차차...)
    text = re.sub(r'(.)\1{4,}', '', text)
    # 연속 공백 → 단일 공백
    text = re.sub(r' {2,}', ' ', text)
    # 연속 빈줄 → 단일 빈줄
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()




def chunk_from_alldata(doc_name: str, all_data: dict, size: int = 800) -> list[str] | None:
    """ALL_DATA에서 문서별 청크 생성. Docling/기존 포맷 자동 감지."""
    if doc_name not in all_data:
        return None

    data = all_data[doc_name]

    # Docling 포맷 감지: 리스트 + 각 항목에 "content" 키
    if isinstance(data, list) and len(data) > 0 and "content" in data[0]:
        return _chunk_docling(data, size)
    
    # 기존 포맷: {"metadata": [...]}
    if isinstance(data, dict) and "metadata" in data:
        return _chunk_legacy(data, size)

    return None


def _chunk_legacy(data: dict, size: int = 800) -> list[str]:
    """기존 포맷 청킹 (원래 chunk_from_alldata 로직)."""
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
            parts.append(clean_text(page["text"]))

        if page.get("table"):
            for t in page["table"]:
                table_text = format_table(t)
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


# v5
def _chunk_docling(items: list[dict], size: int = 800) -> list[str]:
    """Docling 포맷 의미 기반 청킹: section_path + 표 복원."""

    # 1) 표 복원: table_id별로 행을 합침
    tables = {}
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("type") == "table_row" and meta.get("table_id") is not None:
            tables.setdefault(meta["table_id"], []).append(item)

    table_blocks = {}
    for tid, rows in tables.items():
        rows.sort(key=lambda x: x.get("metadata", {}).get("row", 0))
        combined = "\n".join(r.get("content", "") for r in rows)
        table_blocks[tid] = {
            "content": f"[표]\n{combined}",
            "metadata": rows[0].get("metadata", {})
        }

    # 2) 원본 순서 유지하면서 table_row → table_block 치환
    all_items = []
    seen_tables = set()
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("type") == "table_row":
            tid = meta.get("table_id")
            if tid not in seen_tables and tid in table_blocks:
                seen_tables.add(tid)
                all_items.append(table_blocks[tid])
        else:
            all_items.append(item)

    # 3) 의미 단위(segment) 구성: section_path가 바뀌면 새 세그먼트
    segments = []  # [(section_label, text)]
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
        sec2 = meta.get("section_level_2", "") or ""

        # 노이즈 필터링
        if len(content) <= 3 and label != "section_header":
            continue

        # section_path 전환 → 새 세그먼트
        if path != current_path and current_parts:
            segments.append("\n".join(current_parts))
            current_parts = []
        current_path = path

        # section_header는 제목으로 표시
        if label == "section_header":
            current_parts.append(f"\n## {content} (p.{page})")
        else:
            current_parts.append(clean_text(content))

    if current_parts:
        segments.append("\n".join(current_parts))

    # 4) 세그먼트 → 청크: 작은 건 합치고, 큰 건 분할
    chunks = []
    buffer = ""

    for seg in segments:
        # 버퍼 + 현재 세그먼트가 size 이내면 합침
        if len(buffer) + len(seg) + 1 <= size:
            buffer = f"{buffer}\n{seg}" if buffer else seg
        else:
            # 버퍼에 내용 있으면 저장
            if buffer:
                chunks.append(buffer.strip())
            # 현재 세그먼트가 size보다 크면 분할
            if len(seg) > size:
                for i in range(0, len(seg), size):
                    chunks.append(seg[i:i + size].strip())
                buffer = ""
            else:
                buffer = seg

    if buffer:
        chunks.append(buffer.strip())

    return chunks





def show_sample(docs):
    test_name = docs[0].name
    chunks_test = chunk_from_alldata(test_name, ALL_DATA)

    if chunks_test is None:
        text = clean_text(extract_text(docs[0]))
        chunks_test = chunk(text)

    print(f"문서: {test_name}")
    print(f"총 청크 수: {len(chunks_test)}")

    for i in range(min(5, len(chunks_test))):
        print(f"\n=== 청크 {i} ({len(chunks_test[i])}자) ===")
        print(chunks_test[i])
