from pathlib import Path

BASE_DIR = Path.cwd().parent  # /codeit-part3-team4
RAW_FOLDER = BASE_DIR / "data/raw/files"  # 수정된 경로
DATA_FOLDER = BASE_DIR / "data"  # 수정된 경로


# 폴더에서 PDF 목록 가져오기
folder = Path(RAW_FOLDER)
pdf_paths = [p for p in folder.glob("*.pdf")]
docs = sorted(pdf_paths)