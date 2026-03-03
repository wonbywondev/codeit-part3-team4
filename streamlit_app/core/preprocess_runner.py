from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

from .config import AppPaths
from .text_clean import post_clean_text


def _get_preprocess_dir(paths: AppPaths) -> Path:
    # 팀 구조: repo_root/notebooks/preprocess
    return paths.repo_root / "notebooks" / "preprocess"


def detect_available_pp_versions(paths: AppPaths) -> List[str]:
    preprocess_dir = _get_preprocess_dir(paths)
    if not preprocess_dir.exists():
        return []
    versions = [p.stem for p in preprocess_dir.glob("pp_v*.py")]
    return sorted(versions)


def run_preprocessing(
    version: str,
    doc_id: str,
    size: int,
    paths: AppPaths,
) -> Tuple[bool, str]:
    """
    notebooks/preprocess/pp_vX 를 동적으로 import하여
    doc_id에 대한 청크를 생성하고
    cache/chunks/<version>/<doc_id>.jsonl 로 저장

    pp 파일은 건드리지 않고, export 시점에 텍스트만 정규화/중복축약한다.
    """
    try:
        import sys
        notebooks_dir = paths.repo_root / "notebooks"
        if str(notebooks_dir) not in sys.path:
            sys.path.append(str(notebooks_dir))

        module = importlib.import_module(f"preprocess.{version}")

        chunks_out = []

        # v6: chunk_records_from_alldata 존재 :contentReference[oaicite:0]{index=0}
        if hasattr(module, "chunk_records_from_alldata"):
            records = module.chunk_records_from_alldata(doc_id, size=size)
            if records is None:
                return False, f"[{version}] ALL_DATA에서 문서를 찾지 못함: {doc_id}"

            for i, r in enumerate(records):
                meta = r.get("metadata", {}) or {}
                page = meta.get("page", 0)
                try:
                    page = int(page)
                except Exception:
                    page = 0

                content = post_clean_text(str(r.get("content", "") or ""))
                if not content:
                    continue

                chunks_out.append({
                    "doc_id": doc_id,
                    "page": page,
                    "chunk_id": f"{doc_id}_{i}",
                    "text": content,
                    "section_path": str(meta.get("section_path", "") or ""),
                })

        # v4/v5: chunk_from_alldata만 존재할 수 있음 
        elif hasattr(module, "chunk_from_alldata"):
            raw_chunks = module.chunk_from_alldata(doc_id, size=size)
            if raw_chunks is None:
                return False, f"[{version}] ALL_DATA에서 문서를 찾지 못함: {doc_id}"

            for i, text in enumerate(raw_chunks):
                content = post_clean_text(str(text or ""))
                if not content:
                    continue
                # page 메타가 없는 버전은 1로 두고, UI에서 페이지 미리보기는 fallback(pdf_fallback) 권장
                chunks_out.append({
                    "doc_id": doc_id,
                    "page": 1,
                    "chunk_id": f"{doc_id}_{i}",
                    "text": content,
                    "section_path": "",
                })

        else:
            return False, f"[{version}] 지원하는 chunk 함수가 없습니다."

        out_dir = paths.chunks_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_id}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks_out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        return True, f"저장 완료 → {out_path} (chunks={len(chunks_out)})"

    except Exception:
        return False, f"전처리 실패:\n{traceback.format_exc()}"