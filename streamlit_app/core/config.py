from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# streamlit_app/core/config.py -> parents[2] == repo root (codeit-part3-team4)
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")  # ✅ /repo/.env 자동 로드

@dataclass(frozen=True)
class AppPaths:
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    pdf_dir: Path = REPO_ROOT / "data" / "raw" / "files"
    eval_dir: Path = REPO_ROOT / "data" / "raw" / "eval"

    cache_dir: Path = REPO_ROOT / "cache"
    index_dir: Path = REPO_ROOT / "cache" / "indices"
    chunks_dir: Path = REPO_ROOT / "cache" / "chunks"  # ✅ 전처리 chunk 산출물 추천 위치

    # 없으면 자동으로 fallback
    text_metadata_path: Path = REPO_ROOT / "data" / "raw" / "03_text_metadata.json"
    index_pages_path: Path = REPO_ROOT / "data" / "raw" / "01_index_pages.json"  # 선택
    section_words_path: Path = REPO_ROOT / "data" / "raw" / "02_section_words.json"  # 선택

@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")