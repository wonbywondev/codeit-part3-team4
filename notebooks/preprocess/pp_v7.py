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

from pp_basic import BASE_DIR, RAW_DIR, docs

import re
import os
import json
from pathlib import Path
import pdfplumber
import faiss
from paddleocr import PaddleOCR, PaddleOCRVL, doc


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
    ALL_DATA = _raw

doc = docs[0]
PaddleOCR.ocr()