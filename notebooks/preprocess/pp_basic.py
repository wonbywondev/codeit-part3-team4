import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw/files"  # 수정된 경로
DATA_DIR = BASE_DIR / "data"  # 수정된 경로
EVAL_DIR = BASE_DIR / "data/raw/eval"

GOLD_EVIDENCE_CSV = EVAL_DIR / "gold_evidence.csv"
GOLD_FIELDS_JSONL = EVAL_DIR / "gold_fields.jsonl"


# 폴더에서 PDF 목록 가져오기
folder = Path(RAW_DIR)
pdf_paths = [p for p in folder.glob("*.pdf")]
docs = sorted(pdf_paths)


# index_pages_path = os.path.join(BASE_DIR, "data", "01_index_pages.json")

# with open(index_pages_path, "r", encoding="utf-8") as f:
#     index_pages = json.load(f)