# streamlit_app/core/runtime_imports.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

from .config import AppPaths

def ensure_notebooks_on_syspath(paths: AppPaths) -> None:
    """
    repo_root/notebooks 를 sys.path에 올려서
    notebooks/preprocess/* 모듈을 import 가능하게 만듦.
    """
    notebooks_dir = paths.repo_root / "notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.append(str(notebooks_dir))

def list_preprocess_modules(paths: AppPaths, pattern: str) -> List[str]:
    """
    notebooks/preprocess 아래에서 pattern에 맞는 파일(stem) 목록
    """
    preprocess_dir = paths.repo_root / "notebooks" / "preprocess"
    if not preprocess_dir.exists():
        return []
    return sorted([p.stem for p in preprocess_dir.glob(pattern)])