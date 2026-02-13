import os
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw/files_30"  # 평가용 30개 파일
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "data/raw/eval"
GOLD_EVIDENCE_CSV = EVAL_DIR / "gold_evidence.csv"
GOLD_FIELDS_JSONL = EVAL_DIR / "gold_fields.jsonl"
# 폴더에서 PDF 목록 가져오기
folder = Path(RAW_DIR)
pdf_paths = [p for p in folder.glob("*.pdf")]
docs = sorted(pdf_paths)
