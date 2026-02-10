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
"""

from preprocess.pp_basic import BASE_DIR, RAW_DIR, docs

import re
import os
import json
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

version = "4"


all_data_path = os.path.join(BASE_DIR, "data", f"ALL_DATA_v{version}.json")
index_pages_path = os.path.join(BASE_DIR, "data", "01_index_pages.json")

with open(all_data_path, "r", encoding="utf-8") as f:
    ALL_DATA = json.load(f)

with open(index_pages_path, "r", encoding="utf-8") as f:
    index_pages = json.load(f)


# 기본 함수
# Chunking
def chunk(text: str, size: int = 800) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]


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


# v1
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


# v2
def chunk_from_alldata(doc_name: str, all_data: dict, size: int = 800) -> list[str] | None:
    """ALL_DATA의 섹션 단위로 그룹핑 후 size 단위로 분할."""
    if doc_name not in all_data:
        return None

    pages = all_data[doc_name]["metadata"]

    # 1) 섹션별로 페이지 텍스트 그룹핑
    sections = []  # [(섹션명, 텍스트)]
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

        # 섹션이 바뀌면 이전 섹션 저장
        if sec_str != current_sec and current_parts:
            sections.append((current_sec, "\n".join(current_parts)))
            current_parts = []
        current_sec = sec_str

        # 페이지 텍스트 조립
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

    # 마지막 섹션
    if current_parts:
        sections.append((current_sec, "\n".join(current_parts)))

    # 2) 섹션별로 size 단위 분할 (섹션 경계에서 끊김 방지)
    chunks = []
    for sec_name, sec_text in sections:
        for i in range(0, len(sec_text), size):
            chunks.append(sec_text[i:i + size])

    return chunks


# n.3부터 쓰임
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
