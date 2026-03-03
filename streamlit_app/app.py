# streamlit_app/app.py
from __future__ import annotations

import sys
import json
import re
import time
import pickle
import hashlib
import streamlit.components.v1 as components
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

from core.config import AppPaths, AppConfig
from core.config_io import write_fixed_config_py
from core.loaders import make_pdf_map, get_chunks
from core.preprocess_runner import detect_available_pp_versions, run_preprocessing
from core.llm import summarize_with_evidence
from core.render import (
    get_pdf_num_pages,
    render_pdf_page_png,
    render_pdf_page_png_with_highlights,
    find_pages_with_hit_counts,
    find_pages_for_queries,
)

# =========================================================
# Path bootstrap: notebooks on sys.path
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

# =========================================================
# Init
# =========================================================
paths = AppPaths()
cfg = AppConfig()

CACHE_DIR = paths.cache_dir
SERVICE_CFG_PATH = CACHE_DIR / "service_config.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIXED_CONFIG_PY = paths.repo_root / "streamlit_app" / "core" / "fixed_config.py"

try:
    from core.fixed_config import CONFIG as FIXED_DEFAULT
except Exception:
    FIXED_DEFAULT = {}

BASE_DEFAULT_CFG: Dict[str, Any] = {
    # data
    "service_chunk_mode": "runtime_c1",  # precomputed | runtime_c1
    "pp_version": "pp_v6",
    "chunk_length": 800,

    # retriever
    "service_retriever_mode": "R2",  # R1/R2/R3
    "service_retrieve_k": 20,
    "service_context_k": 16,

    # R3
    "alpha": 0.7,
    "bm25_candidates": 300,

    # embed
    "embed_model_name": "nlpai-lab/KoE5",
    "embed_batch_size": 64,
    "embed_device": "cuda",

    # LLM
    "max_context_chars": 2500,
    "max_completion_tokens": 800,
    "temperature": 0.1,
    "reasoning_effort": "low",
    "generator_retries": 1,
    "fallback_model": "gpt-4.1-mini",

    # pdf scan
    "enable_pdf_scan": True,
    "confirmed_max": 2,
    "candidate_max": 12,
    "max_pages_scan": 160,

    # highlight accuracy (핵심) - 기본값을 살짝 “관대”하게
    "hl_window": 70,
    "hl_max_phrases": 8,
    "hl_min_token_len": 3,
    "hl_max_token": 10,
    "hl_min_score": 1,

    # highlight fallback
    "enable_hl_fallback_keywords": True,

    # compatibility
    "top_k": 16,
}
DEFAULT_CFG = {**BASE_DEFAULT_CFG, **(FIXED_DEFAULT or {})}

WEAK_KEYWORDS = [
    "예산", "사업비", "소요예산", "부가가치세", "VAT", "계약", "입찰",
    "기간", "마감", "제출", "기관", "요구사항"
]

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="RAG Service (KoE5 + FAISS cosine)", layout="wide")
st.title("🤖 RAG 서비스")

if not paths.pdf_dir.exists():
    st.error(f"PDF 폴더가 없습니다: {paths.pdf_dir}")
    st.stop()

pdf_map = make_pdf_map(paths.pdf_dir)
pdf_names = list(pdf_map.keys())
pp_versions = detect_available_pp_versions(paths) or ["pp_v5", "pp_v6", "pp_v4"]

tab_service, tab_settings = st.tabs(["🟢 서비스(채팅)", "⚙️ 설정(서비스)"])

# =========================================================
# Config IO
# =========================================================
def load_cfg() -> Dict[str, Any]:
    def _migrate_legacy_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data or {})
        # 이전 버전 키 호환: highlight_* -> hl_*
        if "hl_window" not in d and "highlight_snip_len" in d:
            d["hl_window"] = d["highlight_snip_len"]
        if "hl_max_phrases" not in d and "highlight_pick_top" in d:
            d["hl_max_phrases"] = d["highlight_pick_top"]
        if "hl_max_token" not in d and "highlight_max_candidates" in d:
            d["hl_max_token"] = d["highlight_max_candidates"]
        return d

    if SERVICE_CFG_PATH.exists():
        try:
            data = json.loads(SERVICE_CFG_PATH.read_text(encoding="utf-8"))
            data = _migrate_legacy_keys(data)
            return {**DEFAULT_CFG, **data}
        except Exception:
            return DEFAULT_CFG.copy()
    return DEFAULT_CFG.copy()

def save_cfg(cfg_obj: Dict[str, Any]) -> None:
    SERVICE_CFG_PATH.write_text(json.dumps(cfg_obj, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================================================
# Helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _norm_loose(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) + eps)
        return (x / denom).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / denom).astype(np.float32)

def _resolve_embed_device(requested: str) -> str:
    req = (requested or "cpu").lower().strip()
    if req == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"

def start_elapsed_timer(label: str = "검색/답변 생성 중..."):
    ph = st.empty()
    t0 = time.perf_counter()
    running = {"on": True}

    def tick():
        if running["on"]:
            ph.caption(f"⏳ {label}  {time.perf_counter() - t0:.1f}s")

    def stop():
        running["on"] = False
        ph.caption(f"✅ 완료  {time.perf_counter() - t0:.1f}s")

    tick()
    return {"tick": tick, "stop": stop, "t0": t0}

# =========================================================
# Caches: chunks / model
# =========================================================
@st.cache_data(show_spinner=False)
def cached_runtime_c1_chunks(doc_id: str, pdf_path: str, chunk_length: int) -> tuple[list, str]:
    chunks, artifact_path = get_chunks(
        doc_id=doc_id,
        pdf_path=pdf_path,
        paths=paths,
        source="runtime_c1",
        chunk_length=int(chunk_length),
    )
    return chunks, str(artifact_path)

@st.cache_data(show_spinner=False)
def cached_precomputed_chunks(doc_id: str, pdf_path: str, jsonl_path_str: str) -> tuple[list, str]:
    jsonl = Path(jsonl_path_str)
    chunks, artifact_path = get_chunks(
        doc_id=doc_id,
        pdf_path=pdf_path,
        paths=paths,
        source="precomputed_chunks",
        precomputed_jsonl=jsonl,
    )
    return chunks, str(artifact_path)

@st.cache_resource(show_spinner=False)
def cached_embed_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    dev = _resolve_embed_device(device)
    return SentenceTransformer(model_name, device=dev)

# =========================================================
# PDF page text (for accurate highlights)
# =========================================================
@st.cache_data(show_spinner=False)
def cached_page_text(pdf_path: str, page_phys: int) -> str:
    i = int(page_phys) - 1
    if i < 0:
        return ""

    # 1) PyMuPDF 우선
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        if i >= doc.page_count:
            doc.close()
            return ""
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        doc.close()
        return _norm_loose(txt)
    except Exception:
        pass

    # 2) pdfplumber fallback
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if i >= len(pdf.pages):
                return ""
            txt = pdf.pages[i].extract_text() or ""
            return _norm_loose(txt)
    except Exception:
        return ""

# =========================================================
# Disk cache helpers (BM25 / FAISS)
# =========================================================
def _artifact_signature(path: Path) -> str:
    stt = path.stat()
    raw = f"{path.resolve()}||{stt.st_mtime_ns}||{stt.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

def _svc_cache_dir() -> Path:
    d = paths.cache_dir / "service_retrievers"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _cache_paths(doc_id: str, source_key: str, sig: str, model_name: str) -> Dict[str, Path]:
    safe = doc_id.replace("/", "_")
    base = f"{safe}__{source_key}__sig={sig}__m={model_name.replace('/','_')}__ix=flatip_cos"
    d = _svc_cache_dir()
    return {
        "bm25": d / f"{base}.bm25.pkl",
        "faiss": d / f"{base}.faiss",
        "embs": d / f"{base}.embs.npy",
    }

def build_or_load_bm25(doc_id: str, source_key: str, sig: str, texts: List[str]):
    from rank_bm25 import BM25Okapi
    p = _cache_paths(doc_id, source_key, sig, "bm25")["bm25"]
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(p, "wb") as f:
        pickle.dump(bm25, f)
    return bm25

def build_or_load_faiss(
    doc_id: str,
    source_key: str,
    sig: str,
    texts: List[str],
    model_name: str,
    batch_size: int,
    embed_device: str,
):
    import faiss
    ps = _cache_paths(doc_id, source_key, sig, model_name)
    faiss_path, npy_path = ps["faiss"], ps["embs"]

    if faiss_path.exists() and npy_path.exists():
        index = faiss.read_index(str(faiss_path))
        embs = np.load(npy_path)
        return index, embs

    model = cached_embed_model(model_name, embed_device)
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=int(batch_size),
    ).astype("float32")

    embs = _l2_normalize(embs)

    index = faiss.IndexFlatIP(int(embs.shape[1]))
    index.add(embs)

    faiss.write_index(index, str(faiss_path))
    np.save(npy_path, embs)
    return index, embs

# =========================================================
# Index pages
# =========================================================
@st.cache_data(show_spinner=False)
def load_index_pages(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

_index_pages = load_index_pages(str(paths.index_pages_path))

def get_start_page(doc_id: str) -> int:
    info = _index_pages.get(doc_id, {})
    return max(1, int(info.get("start_page_label", 1)))

def get_index_page_labels(doc_id: str) -> List[int]:
    info = _index_pages.get(doc_id, {})
    return [int(p) for p in info.get("index_page_label", [])]

def get_front_pages(doc_id: str) -> List[tuple]:
    start = get_start_page(doc_id)
    if start <= 1:
        return []
    index_labels = set(get_index_page_labels(doc_id))
    result = []
    for p in range(1, start):
        if p == 1:
            lbl = "표지"
        elif p in index_labels:
            lbl = "목차"
        else:
            lbl = "앞부분"
        result.append((p, lbl))
    return result

def get_banned_pages_for_routing(doc_id: str, pdf_path: str, start_page: int, n_pages: int) -> List[int]:
    banned = set(get_index_page_labels(doc_id))
    try:
        toc_hits = find_pages_for_queries(
            pdf_path,
            ["목차", "contents", "CONTENTS"],
            start_page=1,
            max_pages_scan=min(14, int(n_pages)),
        )
    except Exception:
        toc_hits = []
    for p in toc_hits:
        pi = int(p)
        if 1 <= pi <= min(int(n_pages), int(start_page) + 12):
            banned.add(pi)
    return sorted(banned)

# =========================================================
# Session state
# =========================================================
def ensure_chat_state():
    if "doc_messages" not in st.session_state:
        st.session_state["doc_messages"] = {}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "active_doc" not in st.session_state:
        st.session_state["active_doc"] = None

def switch_doc(doc_id: str):
    ensure_chat_state()
    prev = st.session_state.get("active_doc")
    if prev != doc_id:
        if prev is not None:
            st.session_state["doc_messages"][prev] = st.session_state.get("messages", [])
        st.session_state["messages"] = st.session_state["doc_messages"].get(doc_id, [])
        st.session_state["active_doc"] = doc_id

        for k in [
            "svc_render_phys",
            "svc_confirmed_pages",
            "svc_candidate_pages",
            "svc_highlight_queries",
            "svc_pending_phys",
            "svc_pending_apply",
            "svc_pv_page_input",
            "svc_prepared_sig",
        ]:
            st.session_state.pop(k, None)

# =========================================================
# Text helpers
# =========================================================
def clean_answer_remove_page_lines(answer: str) -> str:
    a = answer or ""
    a = re.sub(r"(?m)^\s*근거\s*페이지\s*:\s*.*$", "", a)
    a = re.sub(r"(?m)^\s*근거\s*페이지\s*.*$", "", a)
    a = re.sub(r"\n{3,}", "\n\n", a).strip()
    return a

def _extract_recent_history(messages: List[Dict[str, str]], max_msgs: int = 6) -> str:
    if not messages:
        return ""
    rows: List[str] = []
    for m in messages[-max_msgs:]:
        role = "사용자" if m.get("role") == "user" else "어시스턴트"
        content = _norm(m.get("content", ""))
        if content:
            rows.append(f"{role}: {content}")
    return "\n".join(rows)

_BUDGET_EXCLUDE_PATTERNS = [
    "총사업예산",
    "총 사업예산",
    "제안서 보상",
    "낙찰자로 결정되지",
    "소프트웨어사업 계약",
    "관리감독 지침",
    "20억원 이상",
    "20억 원 이상",
    "20억원 미만의 소프트웨어사업",
    "20억 원 미만의 소프트웨어사업",
    "보상 대상에서 제외",
]

def _is_budget_noise_line(line: str) -> bool:
    ln = _norm_loose(line)
    if not ln:
        return True
    return any(p in ln for p in _BUDGET_EXCLUDE_PATTERNS)

def _parse_money_to_won(text: str) -> int:
    t = _norm(text)
    cands: List[int] = []

    for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+)\s*원\b", t):
        try:
            cands.append(int(m.group(1).replace(",", "")))
        except Exception:
            pass

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*억(?:\s*(\d+(?:\.\d+)?)\s*만)?", t):
        try:
            eok = float(m.group(1))
            man = float(m.group(2)) if m.group(2) else 0.0
            cands.append(int(eok * 100_000_000 + man * 10_000))
        except Exception:
            pass
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*만\s*원?", t):
        try:
            cands.append(int(float(m.group(1)) * 10_000))
        except Exception:
            pass

    for m in re.finditer(r"\b(\d{4,})\s*원\b", t):
        try:
            cands.append(int(m.group(1)))
        except Exception:
            pass

    if not cands:
        return 0
    return max(cands)

def _format_won(n: int) -> str:
    return f"{int(n):,}원"

def _extract_budget_threshold(question: str) -> Tuple[str, int]:
    q = _norm(question)
    thr = _parse_money_to_won(q)
    if thr <= 0:
        return "", 0
    if any(k in q for k in ["미만", "<"]):
        return "<", thr
    if any(k in q for k in ["이하", "≤", "<="]):
        return "<=", thr
    if any(k in q for k in ["초과", ">", "넘"]):
        return ">", thr
    if any(k in q for k in ["이상", "≥", ">="]):
        return ">=", thr
    return "", 0

def _extract_budget_fact_from_ctx(chunks: list, ctx_idxs: List[int]) -> Tuple[int, int, str]:
    best = (0, 0, "", -10.0)
    for i in ctx_idxs:
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        txt = _norm_loose(getattr(c, "text", ""))
        if not txt:
            continue
        parts = re.split(r"(?:\n+|[.!?。]\s+|다\.\s+)", txt)
        for ln in parts:
            ln_n = _norm_loose(ln)
            if not ln_n:
                continue
            if not any(k in ln_n for k in ["예산", "사업비", "소요예산"]):
                continue
            if _is_budget_noise_line(ln_n):
                continue
            won = _parse_money_to_won(ln_n)
            if won <= 0:
                continue

            score = 0.0
            if "소요예산" in ln_n:
                score += 3.0
            if "사업예산" in ln_n:
                score += 2.0
            if "부가가치세" in ln_n or "VAT" in ln_n:
                score += 1.5
            if re.search(r"\b\d[\d,]{3,}\s*원", ln_n):
                score += 2.0
            if int(getattr(c, "page", 999) or 999) <= 20:
                score += 0.8

            if score > best[3]:
                best = (won, int(getattr(c, "page", 0) or 0), ln_n[:120], score)

    if best[0] > 0 and best[3] >= 2.0:
        return best[0], best[1], best[2]
    return 0, 0, ""

def _budget_page_quality(pdf_path: str, page: int) -> float:
    txt = _norm_loose(cached_page_text(pdf_path, int(page)))
    if not txt:
        return 0.0
    score = 0.0
    if "소요예산" in txt:
        score += 3.0
    if "사업예산" in txt:
        score += 2.0
    if "부가가치세" in txt or "VAT" in txt:
        score += 1.0
    if re.search(r"\b\d[\d,]{3,}\s*원", txt):
        score += 1.5
    if any(p in txt for p in _BUDGET_EXCLUDE_PATTERNS):
        score -= 4.0
    return score

def _name_page_quality(pdf_path: str, page: int) -> float:
    txt = _norm_loose(cached_page_text(pdf_path, int(page)))
    if not txt:
        return 0.0
    txt_ns = re.sub(r"[^가-힣A-Za-z0-9]", "", txt)
    score = 0.0
    if "목차" in txt or "contents" in txt.lower():
        score -= 6.0
    if "사업명" in txt or "사업명" in txt_ns:
        score += 3.0
    if "용역사업" in txt or "용역사업" in txt_ns:
        score += 1.5
    if re.search(r"사\s*업\s*명\s*[:：]", txt):
        score += 2.0
    if re.search(r"\[?\s*사\s*업\s*명\s*\]?", txt):
        score += 1.0
    if any(k in txt for k in ["일반사항", "유의사항", "제안서 작성"]):
        score -= 1.8
    return score

def _name_page_priors(
    pdf_path: str,
    *,
    start_page: int,
    n_pages: int,
    banned_pages: set[int],
    max_scan_pages: int = 28,
) -> List[int]:
    if n_pages <= 0:
        return []
    end_page = min(int(n_pages), int(start_page) + max_scan_pages - 1)
    scored: List[Tuple[float, int]] = []
    for p in range(int(start_page), end_page + 1):
        if p in banned_pages:
            continue
        txt = _norm_loose(cached_page_text(pdf_path, p))
        if not txt:
            continue
        txt_ns = re.sub(r"[^가-힣A-Za-z0-9]", "", txt)
        score = 0.0
        if "사업명" in txt or "사업명" in txt_ns:
            score += 3.0
        if re.search(r"사\s*업\s*명\s*[:：]", txt):
            score += 2.0
        if "용역사업" in txt or "용역사업" in txt_ns:
            score += 1.2
        if "사업개요" in txt or "제안요청서" in txt:
            score += 0.6
        if "목차" in txt or "contents" in txt.lower():
            score -= 8.0
        score -= max(0, p - int(start_page)) * 0.08
        if any(k in txt for k in ["일반사항", "유의사항", "제안서 작성"]):
            score -= 1.6
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _s, p in scored]

def _extract_name_phrases_from_answer(answer: str) -> List[str]:
    a = _norm(answer)
    out: List[str] = []
    quoted = re.findall(r'"([^"]+)"', a)
    for q in quoted:
        qq = _norm(q)
        if len(qq) >= 6:
            out.append(qq)
    m = re.search(r"사업명은\s*[:：]?\s*([^.\n]+)", a)
    if m:
        seg = _norm(m.group(1))
        if len(seg) >= 6:
            out.append(seg)
    uniq, seen = [], set()
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:3]

def _name_answer_match_pages(
    pdf_path: str,
    answer: str,
    *,
    start_page: int,
    n_pages: int,
    banned_pages: set[int],
    max_scan_pages: int = 40,
) -> List[int]:
    phrases = _extract_name_phrases_from_answer(answer)
    if not phrases:
        return []
    end_page = min(int(n_pages), int(start_page) + max_scan_pages - 1)
    scored: List[Tuple[float, int]] = []
    for p in range(int(start_page), end_page + 1):
        if p in banned_pages:
            continue
        txt = _norm_loose(cached_page_text(pdf_path, p))
        if not txt:
            continue
        txt_ns = re.sub(r"[^가-힣A-Za-z0-9]", "", txt)
        s = 0.0
        if "사업명" in txt or "사업명" in txt_ns:
            s += 2.0
        for ph in phrases:
            phn = _norm_loose(ph)
            if not phn:
                continue
            ph_ns = re.sub(r"[^가-힣A-Za-z0-9]", "", phn)
            if phn in txt or (ph_ns and ph_ns in txt_ns):
                s += 4.0
            elif len(phn) > 18 and (phn[:18] in txt or (ph_ns and ph_ns[:18] in txt_ns)):
                s += 2.5
        if "목차" in txt:
            s -= 8.0
        s -= max(0, p - int(start_page)) * 0.08
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _s, p in scored]

# =========================================================
# Evidence page helpers
# =========================================================
def provisional_pages_from_idxs(chunks: list, idxs: List[int], max_pages: int = 3) -> List[int]:
    pages = []
    for i in idxs[: max(10, max_pages * 3)]:
        try:
            p = int(getattr(chunks[int(i)], "page", 0) or 0)
        except Exception:
            p = 0
        if p > 0 and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def detect_query_intent(question: str) -> str:
    q = _norm(question)
    if any(k in q for k in ["예산", "사업비", "소요예산", "금액"]):
        return "budget"
    if any(k in q for k in ["사업명", "과업명", "용역명", "프로젝트명"]):
        return "name"
    if any(k in q for k in ["기간", "사업기간", "수행기간"]):
        return "period"
    return "generic"

def extract_strong_queries_keywords(question: str, answer: str, top_text: str) -> List[str]:
    q = _norm(question)
    a = _norm(answer)
    out: List[str] = []

    quoted = re.findall(r'"([^"]+)"', a)
    for s in quoted[:2]:
        s = _norm(s)
        if len(s) >= 6:
            out.append(s[:80])

    if ":" in a:
        tail = _norm(a.split(":")[-1])[:80]
        if len(tail) >= 10:
            out.append(tail)

    lines = [x.strip() for x in (top_text or "").replace("\r", "\n").split("\n") if x.strip()]
    for kw in WEAK_KEYWORDS:
        if kw in q:
            for ln in lines:
                if kw in ln and len(ln) >= 8:
                    out.append(_norm(ln)[:80])
                    break
            break

    nums = re.findall(r"\b\d[\d,]{3,}\b", a)
    if nums:
        out.append(nums[0])
        out.append(nums[0].replace(",", ""))

    if not out and a:
        out.append(a[:60])

    uniq, seen = [], set()
    for x in out:
        x = _norm(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:6]

def pick_pages_confirmed_candidate(
    pdf_path: str,
    strong_queries: List[str],
    weak_queries: List[str],
    cfg_obj: Dict[str, Any],
    fallback_pages: List[int],
    question: str = "",
    banned_pages: List[int] | None = None,
    n_pages: int = 0,
    start_page: int = 1,
) -> Tuple[List[int], List[int]]:
    confirmed_max = int(cfg_obj["confirmed_max"])
    candidate_max = int(cfg_obj["candidate_max"])
    max_pages_scan = int(cfg_obj["max_pages_scan"])

    confirmed: List[int] = []
    candidate: List[int] = []

    local_pages: List[int] = []
    if fallback_pages:
        seen = set()
        for p in fallback_pages:
            pi = int(p)
            for c in range(pi - 12, pi + 13):
                if c >= start_page and c not in seen:
                    seen.add(c)
                    local_pages.append(c)

    banned = set(int(p) for p in (banned_pages or []) if int(p) >= 1)

    hit_map = {}
    if strong_queries:
        try:
            hit_map = find_pages_with_hit_counts(
                pdf_path, strong_queries,
                start_page=start_page, max_pages_scan=max_pages_scan,
                page_whitelist=(local_pages or None),
            )
            if not hit_map and local_pages:
                hit_map = find_pages_with_hit_counts(
                    pdf_path, strong_queries,
                    start_page=start_page, max_pages_scan=max_pages_scan,
                )
        except Exception:
            hit_map = {}

    intent = detect_query_intent(question)
    if hit_map:
        priors = [int(p) for p in (fallback_pages or []) if int(p) >= int(start_page)]
        if priors:
            def _rank_key(item: Tuple[int, int]) -> Tuple[float, int]:
                page, hits = int(item[0]), int(item[1])
                best_dist = min(abs(page - p) for p in priors)
                prior_boost = 3.0 if page in priors else 0.0
                dist_penalty = min(1.2, best_dist / 15.0)
                score = float(hits) + prior_boost - dist_penalty
                if intent == "name":
                    score -= float(page) * 0.03
                    score += _name_page_quality(pdf_path, page)
                elif intent == "budget":
                    score += 0.8 if page <= (start_page + 20) else 0.0
                    score += _budget_page_quality(pdf_path, page)
                if page in banned:
                    score -= 8.0
                return (score, -page)
            ranked = sorted(hit_map.items(), key=_rank_key, reverse=True)
        else:
            ranked = sorted(hit_map.items(), key=lambda kv: (-kv[1], kv[0]))

        confirmed = [p for p, _ in ranked[:confirmed_max]]
        candidate = [p for p, _ in ranked[confirmed_max:confirmed_max + candidate_max]]
    else:
        confirmed = (fallback_pages or [])[:confirmed_max]

    if weak_queries:
        try:
            extra = find_pages_for_queries(
                pdf_path, weak_queries,
                start_page=start_page, max_pages_scan=max_pages_scan,
                page_whitelist=(local_pages or None),
            )
            if not extra and local_pages:
                extra = find_pages_for_queries(
                    pdf_path, weak_queries,
                    start_page=start_page, max_pages_scan=max_pages_scan,
                )
        except Exception:
            extra = []
        for p in extra:
            if p not in confirmed and p not in candidate:
                candidate.append(p)

    if banned:
        confirmed = [p for p in confirmed if p not in banned]
        candidate = [p for p in candidate if p not in banned]
        if not confirmed:
            confirmed = [p for p in (fallback_pages or []) if p not in banned][:confirmed_max]

    if intent == "name":
        name_priors = _name_page_priors(
            pdf_path,
            start_page=int(start_page),
            n_pages=int(n_pages),
            banned_pages=banned,
        )
        prior_name_pages = name_priors
        if not prior_name_pages:
            safe_start = int(start_page)
            if safe_start not in banned:
                prior_name_pages = [safe_start]
            else:
                prior_name_pages = [p for p in range(safe_start, min(safe_start + 6, int(n_pages) + 1)) if p not in banned]
        if not prior_name_pages:
            prior_name_pages = sorted([p for p in (fallback_pages or []) if p not in banned])
        if prior_name_pages:
            confirmed = prior_name_pages[:confirmed_max]
            merged_cand = prior_name_pages[confirmed_max:] + candidate
            uniq_cand, seen = [], set()
            for p in merged_cand:
                if p in confirmed or p in seen:
                    continue
                seen.add(p)
                uniq_cand.append(p)
            candidate = uniq_cand

    candidate = [p for p in candidate if p not in confirmed][:candidate_max]
    return confirmed, candidate

def sync_auto_navigate(target_phys: int, start_page: int, n_pages: int):
    target_phys = max(1, min(int(target_phys), int(n_pages)))
    st.session_state["svc_pending_phys"] = target_phys
    st.session_state["svc_pending_apply"] = True

def _unique_pages_in_order(*page_groups: List[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for group in page_groups:
        for p in group or []:
            pi = int(p)
            if pi <= 0 or pi in seen:
                continue
            seen.add(pi)
            out.append(pi)
    return out

# =========================================================
# Highlight: MAX accuracy
# =========================================================
def extract_key_tokens(question: str, answer: str, cfg_obj: Dict[str, Any]) -> List[str]:
    q = _norm(question)
    a = _norm(answer)

    min_len = int(cfg_obj.get("hl_min_token_len", 3))
    max_tok = int(cfg_obj.get("hl_max_token", 10))

    toks = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", q) if len(t) >= min_len]
    toks = [t for t in toks if t not in WEAK_KEYWORDS][:max_tok]

    nums = re.findall(r"\b\d[\d,]{3,}\b", a)
    num_tokens: List[str] = []
    for n in nums[:2]:
        num_tokens.append(n)
        num_tokens.append(n.replace(",", ""))

    cand = toks + num_tokens
    uniq, seen = [], set()
    for x in cand:
        x = _norm(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def evidence_phrases(evidence: str, snip_len: int = 70, max_n: int = 12) -> List[str]:
    t = (evidence or "").replace("\r", "\n")
    parts = re.split(r"(?:\n+|[.!?。]\s+|다\.\s+)", t)
    sents = [_norm_loose(p) for p in parts if _norm_loose(p)]
    sents = [s for s in sents if len(s) >= 12]

    out = []
    for s in sents[: max_n * 2]:
        out.append(s[:snip_len])
        if len(out) >= max_n:
            break

    uniq, seen = [], set()
    for x in out:
        x = _norm_loose(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:max_n]

def best_page_substrings(page_text: str, tokens: List[str], snip_len: int, max_phrases: int, min_score: int) -> List[str]:
    txt = _norm_loose(page_text)
    if not txt:
        return []

    positions: List[Tuple[int, str]] = []
    for tok in tokens:
        tok = _norm_loose(tok)
        if not tok:
            continue
        start = 0
        while True:
            i = txt.find(tok, start)
            if i == -1:
                break
            positions.append((i, tok))
            start = i + max(1, len(tok))

    if not positions:
        return []

    positions.sort(key=lambda x: x[0])
    candidates: List[Tuple[int, str]] = []
    half = max(10, snip_len // 2)

    for (pos, _tok) in positions[: min(80, len(positions))]:
        a = max(0, pos - half)
        b = min(len(txt), a + snip_len)
        window = txt[a:b].strip()
        if len(window) < 10:
            continue
        score = 0
        for t in tokens:
            t = _norm_loose(t)
            if t and t in window:
                score += 1
        candidates.append((score, window))

    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    picked: List[str] = []
    seen = set()
    for sc, w in candidates:
        if sc < min_score:
            continue
        w = _norm_loose(w)
        if not w or w in seen:
            continue
        seen.add(w)
        picked.append(w)
        if len(picked) >= max_phrases:
            break

    if not picked:
        for sc, w in candidates[:max_phrases]:
            w = _norm_loose(w)
            if w and w not in seen:
                seen.add(w)
                picked.append(w)
        picked = picked[:max_phrases]

    return picked

def build_best_highlights_for_page(
    pdf_path: str,
    page_phys: int,
    question: str,
    answer: str,
    evidence: str,
    cfg_obj: Dict[str, Any],
) -> List[str]:
    snip_len = int(cfg_obj.get("hl_window", 70))
    max_phrases = int(cfg_obj.get("hl_max_phrases", 8))
    min_score = int(cfg_obj.get("hl_min_score", 1))

    page_text = cached_page_text(pdf_path, int(page_phys))
    tokens = extract_key_tokens(question, answer, cfg_obj)

    if not page_text or len(page_text) < 30:
        return [t for t in tokens if t and t not in WEAK_KEYWORDS][:max_phrases]

    ev_ph = evidence_phrases(evidence, snip_len=min(80, snip_len), max_n=10)
    extra_tokens: List[str] = []
    for ph in ev_ph:
        for t in re.findall(r"[가-힣A-Za-z0-9]+", ph):
            if len(t) >= int(cfg_obj.get("hl_min_token_len", 3)) and t not in WEAK_KEYWORDS:
                extra_tokens.append(t)
    extra_tokens = extra_tokens[:12]

    merged_tokens = []
    seen = set()
    for x in (tokens + extra_tokens):
        x = _norm_loose(x)
        if x and x not in seen:
            seen.add(x)
            merged_tokens.append(x)

    picked = best_page_substrings(
        page_text=page_text,
        tokens=merged_tokens,
        snip_len=snip_len,
        max_phrases=max_phrases,
        min_score=min_score,
    )

    num_windows: List[str] = []
    txt = _norm_loose(page_text)
    nums = re.findall(r"\b\d[\d,]{3,}\b", _norm(answer))
    for n in nums[:3]:
        for v in (n, n.replace(",", "")):
            if not v:
                continue
            i = txt.find(v)
            if i == -1:
                continue
            a = max(0, i - max(8, snip_len // 3))
            b = min(len(txt), i + len(v) + max(18, snip_len // 2))
            w = _norm_loose(txt[a:b])
            if len(w) >= 8:
                num_windows.append(w)
    if num_windows:
        merged = []
        seen2 = set()
        for x in (num_windows + picked):
            if x and x not in seen2:
                seen2.add(x)
                merged.append(x)
        picked = merged[:max_phrases]

    return picked

# =========================================================
# Retrieve (R1/R2/R3)
# =========================================================
def retrieve_R1_bm25(bm25, query: str, top_k: int) -> List[int]:
    q = (query or "").strip().split()
    scores = bm25.get_scores(q)
    top = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in top]

def retrieve_R2_flatip(index, embed_model, query: str, top_k: int, batch_size: int) -> List[int]:
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=int(batch_size),
    ).astype("float32")
    q_emb = _l2_normalize(q_emb)
    _, I = index.search(q_emb, min(top_k, index.ntotal))
    return [int(i) for i in I[0]]

def retrieve_R3_hybrid(
    bm25,
    faiss_index,
    embed_model,
    query: str,
    *,
    retrieve_k: int,
    bm25_candidates: int,
    alpha: float,
    batch_size: int,
) -> List[int]:
    q_terms = (query or "").strip().split()
    bm_scores = bm25.get_scores(q_terms)
    cand_n = min(int(bm25_candidates), len(bm_scores))
    cand_idxs = np.argsort(bm_scores)[::-1][:cand_n].astype(int).tolist()

    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=int(batch_size),
    ).astype("float32")
    q_emb = _l2_normalize(q_emb)

    topM = min(max(retrieve_k, cand_n), faiss_index.ntotal)
    _, I = faiss_index.search(q_emb, int(topM))
    vec_idxs = [int(i) for i in I[0]]

    union = list(dict.fromkeys(cand_idxs + vec_idxs))
    rank_map = {idx: (1.0 / (r + 1)) for r, idx in enumerate(vec_idxs)}

    bm_u = np.array([bm_scores[idx] for idx in union], dtype=np.float32)
    if bm_u.size > 0:
        mn, mx = float(bm_u.min()), float(bm_u.max())
        bm_norm = (bm_u - mn) / (mx - mn + 1e-8)
    else:
        bm_norm = bm_u

    scored = []
    for j, idx in enumerate(union):
        vv = rank_map.get(idx, 0.0)
        bb = float(bm_norm[j]) if bm_norm.size > 0 else 0.0
        scored.append((alpha * vv + (1 - alpha) * bb, idx))

    scored.sort(reverse=True)
    return [idx for _s, idx in scored[: min(retrieve_k, len(scored))]]

# =========================================================
# Warmup
# =========================================================
def prepare_doc_assets(doc_id: str, pdf_path: str, cfg_service: Dict[str, Any]) -> str:
    mode = str(cfg_service.get("service_chunk_mode", "runtime_c1"))
    ppver = str(cfg_service.get("pp_version", "pp_v5"))
    chunk_len = int(cfg_service.get("chunk_length", 800))

    if mode == "precomputed":
        jsonl = paths.chunks_dir / ppver / f"{doc_id}.jsonl"
        if not jsonl.exists():
            ok, msg = run_preprocessing(ppver, doc_id, int(chunk_len), paths)
            if not ok:
                raise RuntimeError(f"전처리 생성 실패: {msg}")
        chunks, _artifact_path_str = cached_precomputed_chunks(doc_id, pdf_path, str(jsonl))
        sig_source_path = jsonl
        source_key = f"precomputed__{ppver}__len{chunk_len}"
    else:
        chunks, artifact_path_str = cached_runtime_c1_chunks(doc_id, pdf_path, int(chunk_len))
        sig_source_path = Path(artifact_path_str)
        source_key = f"runtime_c1__len{chunk_len}"

    texts = [c.text for c in chunks]
    sig = _artifact_signature(Path(sig_source_path))

    retriever_mode = str(cfg_service.get("service_retriever_mode", "R2")).upper()
    embed_model_name = str(cfg_service.get("embed_model_name", "nlpai-lab/KoE5"))
    embed_batch_size = int(cfg_service.get("embed_batch_size", 64))
    embed_device = _resolve_embed_device(str(cfg_service.get("embed_device", "cuda")))

    if retriever_mode in ("R1", "R3"):
        _ = build_or_load_bm25(doc_id, source_key, sig, texts)

    if retriever_mode in ("R2", "R3"):
        _ = build_or_load_faiss(doc_id, source_key, sig, texts, embed_model_name, embed_batch_size, embed_device)

    return f"{doc_id}||{source_key}||{sig}||{embed_model_name}"

# =========================================================
# Service tab
# =========================================================
with tab_service:
    cfg_service = load_cfg()
    ensure_chat_state()

    col_chat, col_view = st.columns([1.15, 0.85], gap="large")

    # -------- PDF VIEW --------
    with col_view:
        st.subheader("📄 PDF / 근거 보기")
        doc_id = st.selectbox("문서 선택", pdf_names, key="svc_doc_select")
        pdf_path = str(pdf_map[doc_id])

        switch_doc(doc_id)

        st.markdown("### 🚀 문서 준비")
        if st.button("🚀 문서 준비(인덱스 생성)", use_container_width=True):
            t = start_elapsed_timer("문서 준비 중... (chunks+index)")
            try:
                st.session_state["svc_prepared_sig"] = prepare_doc_assets(doc_id, pdf_path, cfg_service)
            finally:
                t["stop"]()
            st.success("문서 준비 완료!")

        st.caption("✅ 준비됨" if "svc_prepared_sig" in st.session_state else "⏳ 미준비")

        confirmed_pages = st.session_state.get("svc_confirmed_pages", []) or []
        candidate_pages = st.session_state.get("svc_candidate_pages", []) or []
        highlight_queries = st.session_state.get("svc_highlight_queries", []) or []

        start_page = get_start_page(doc_id)
        front_pages = get_front_pages(doc_id)
        offset = start_page - 1

        try:
            n_pages = get_pdf_num_pages(pdf_path)
        except Exception:
            n_pages = 1

        if st.session_state.get("svc_pending_apply"):
            target_phys = int(st.session_state.get("svc_pending_phys", start_page))
            target_phys = max(1, min(target_phys, int(n_pages)))
            st.session_state["svc_render_phys"] = target_phys

            content_max = max(1, n_pages - offset)
            content_p = max(1, min(target_phys - offset, content_max))
            st.session_state["svc_pv_page_input"] = int(content_p)
            st.session_state["svc_pending_apply"] = False

        if front_pages:
            st.caption("앞부분 바로 보기")
            btn_cols = st.columns(len(front_pages))
            for col, (phys_p, lbl) in zip(btn_cols, front_pages):
                if col.button(lbl, key=f"svc_front_{phys_p}"):
                    sync_auto_navigate(phys_p, start_page, n_pages)
                    st.rerun()

        content_max = max(1, n_pages - offset)
        default_content_p = int(st.session_state.get("svc_pv_page_input", 1))
        default_content_p = max(1, min(default_content_p, content_max))
        if "svc_pv_page_input" not in st.session_state:
            st.session_state["svc_pv_page_input"] = int(default_content_p)
        else:
            st.session_state["svc_pv_page_input"] = max(
                1, min(int(st.session_state["svc_pv_page_input"]), content_max)
            )

        def _svc_page_changed():
            target_phys = int(st.session_state["svc_pv_page_input"]) + offset
            st.session_state["svc_render_phys"] = int(target_phys)

        st.number_input(
            f"본문 페이지 (1..{content_max})",
            min_value=1,
            max_value=content_max,
            step=1,
            key="svc_pv_page_input",
            on_change=_svc_page_changed,
        )

        page_to_view = st.session_state.get("svc_render_phys")
        if page_to_view is None:
            page_to_view = int(st.session_state.get("svc_pv_page_input", default_content_p)) + offset
            st.session_state["svc_render_phys"] = int(page_to_view)
        page_to_view = int(page_to_view)

        is_confirmed = page_to_view in confirmed_pages
        is_candidate = page_to_view in candidate_pages

        try:
            if highlight_queries:
                img, hits = render_pdf_page_png_with_highlights(pdf_path, page_to_view, highlight_queries, zoom=2.0)
                tag = "confirmed" if is_confirmed else ("candidate" if is_candidate else "page")
                st.image(img, caption=f"p.{page_to_view - offset} ({tag}, highlights={hits})", width="stretch")
            else:
                img = render_pdf_page_png(pdf_path, page_to_view, zoom=2.0)
                st.image(img, caption=f"p.{page_to_view - offset}", width="stretch")
        except Exception as e:
            st.error("페이지 렌더링 실패")
            st.exception(e)

        if candidate_pages:
            cand_content = [max(1, p - offset) for p in candidate_pages if p >= start_page]
            st.markdown("**🟡 후보 페이지(클릭해서 이동)**")
            row: List[int] = []
            show_list = cand_content[: int(cfg_service.get("candidate_max", 12))]
            for idx, cp in enumerate(show_list):
                row.append(cp)
                if len(row) == 6 or idx == len(show_list) - 1:
                    cols = st.columns(len(row))
                    for c, page_num in zip(cols, row):
                        if c.button(f"{page_num}", key=f"cand_{doc_id}_{page_num}_{idx}"):
                            phys = int(page_num) + offset
                            sync_auto_navigate(phys, start_page, n_pages)
                            st.rerun()
                    row = []
        else:
            st.caption("후보 페이지 없음")

        if confirmed_pages:
            conf_content = [max(1, p - offset) for p in confirmed_pages if p >= start_page]
            st.markdown("**🟢 확정 페이지**")
            st.write(", ".join(str(p) for p in conf_content))

    # -------- CHAT --------
    with col_chat:
            st.subheader("💬 채팅")

            # ✅ ChatGPT 스타일: 아래로 쌓이는 스크롤 채팅 영역
            CHAT_ANCHOR_ID = "chat_end_anchor"

            def _autoscroll_to_bottom(anchor_id: str = CHAT_ANCHOR_ID):
                # rerun마다 실행돼도 무해함
                components.html(
                    f"""
                    <script>
                    const el = window.parent.document.getElementById("{anchor_id}");
                    if (el) {{
                        el.scrollIntoView({{behavior: "smooth", block: "end"}});
                    }}
                    </script>
                    """,
                    height=0,
                )

            chat_box = st.container(height=650)  # 필요하면 높이만 조절

            # 1) 기존 메시지(시간순) 렌더링: 과거 -> 아래로 쌓임
            with chat_box:
                for m in st.session_state.get("messages", []):
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
                        # ✅ 기존 기능 유지: elapsed 표시
                        if m.get("role") == "assistant" and ("elapsed_s" in m):
                            try:
                                st.caption(f"⏱ {float(m['elapsed_s']):.2f}s")
                            except Exception:
                                pass

                # ✅ 스크롤 앵커(항상 맨 아래)
                st.markdown(f'<div id="{CHAT_ANCHOR_ID}"></div>', unsafe_allow_html=True)

            # 2) 입력창은 아래(= ChatGPT처럼)
            user_msg = st.chat_input("질문을 입력하세요")

            # 3) 화면이 최신(맨 아래)로 자동 이동
            _autoscroll_to_bottom(CHAT_ANCHOR_ID)

            # 4) 질문 들어오면: 같은 구조로 (아래에) user -> assistant가 붙고, 완료되면 rerun
            if user_msg:
                api_key = cfg.openai_api_key
                if not api_key:
                    st.error("OPENAI_API_KEY가 없습니다. .env 확인 필요")
                    st.stop()

                # ✅ 기존 로직 유지
                st.session_state["messages"].append({"role": "user", "content": user_msg})

                # ✅ (중요) 답변 생성/로딩도 맨 아래에 보이게: 같은 chat_box 맨 아래에 출력
                with chat_box:
                    with st.chat_message("user"):
                        st.markdown(user_msg)

                    with st.chat_message("assistant"):
                        timer = start_elapsed_timer("검색/답변 생성 중...")  # ✅ 기존 타이머 그대로
                        answer_clean = ""
                        elapsed_s: Optional[float] = None
                        try:
                            # ====== ⬇⬇⬇ 여기부터는 네 기존 답변 생성 로직 "그대로" ======
                            mode = str(cfg_service.get("service_chunk_mode", "runtime_c1"))
                            ppver = str(cfg_service.get("pp_version", "pp_v5"))
                            chunk_len = int(cfg_service.get("chunk_length", 800))

                            # 1) chunks
                            if mode == "precomputed":
                                jsonl = paths.chunks_dir / ppver / f"{doc_id}.jsonl"
                                if not jsonl.exists():
                                    ok, msg = run_preprocessing(ppver, doc_id, int(chunk_len), paths)
                                    if not ok:
                                        st.error("전처리 생성 실패")
                                        st.write(msg)
                                        st.stop()
                                chunks, _artifact_path_str = cached_precomputed_chunks(doc_id, pdf_path, str(jsonl))
                                sig_source_path = jsonl
                                source_key = f"precomputed__{ppver}__len{chunk_len}"
                            else:
                                chunks, artifact_path_str = cached_runtime_c1_chunks(doc_id, pdf_path, int(chunk_len))
                                sig_source_path = Path(artifact_path_str)
                                source_key = f"runtime_c1__len{chunk_len}"

                            texts = [c.text for c in chunks]
                            sig = _artifact_signature(Path(sig_source_path))
                            timer["tick"]()

                            # 2) index
                            retriever_mode = str(cfg_service.get("service_retriever_mode", "R2")).upper()
                            retrieve_k = int(cfg_service.get("service_retrieve_k", 20))
                            context_k = int(cfg_service.get("service_context_k", cfg_service.get("top_k", 16)))
                            if retrieve_k < context_k:
                                retrieve_k = context_k

                            embed_model_name = str(cfg_service.get("embed_model_name", "nlpai-lab/KoE5"))
                            embed_batch_size = int(cfg_service.get("embed_batch_size", 64))
                            embed_device = _resolve_embed_device(str(cfg_service.get("embed_device", "cuda")))

                            bm25 = None
                            faiss_index = None
                            embed_model = None

                            if retriever_mode == "R1":
                                bm25 = build_or_load_bm25(doc_id, source_key, sig, texts)
                            elif retriever_mode == "R2":
                                faiss_index, _ = build_or_load_faiss(
                                    doc_id, source_key, sig, texts, embed_model_name, embed_batch_size, embed_device
                                )
                                embed_model = cached_embed_model(embed_model_name, embed_device)
                            else:
                                bm25 = build_or_load_bm25(doc_id, source_key, sig, texts)
                                faiss_index, _ = build_or_load_faiss(
                                    doc_id, source_key, sig, texts, embed_model_name, embed_batch_size, embed_device
                                )
                                embed_model = cached_embed_model(embed_model_name, embed_device)

                            timer["tick"]()

                            # 3) retrieve
                            if retriever_mode == "R1":
                                idxs = retrieve_R1_bm25(bm25, user_msg, retrieve_k)
                            elif retriever_mode == "R2":
                                idxs = retrieve_R2_flatip(faiss_index, embed_model, user_msg, retrieve_k, embed_batch_size)
                            else:
                                alpha = float(cfg_service.get("alpha", 0.7))
                                bm25_candidates = int(cfg_service.get("bm25_candidates", 300))
                                idxs = retrieve_R3_hybrid(
                                    bm25, faiss_index, embed_model, user_msg,
                                    retrieve_k=retrieve_k,
                                    bm25_candidates=bm25_candidates,
                                    alpha=alpha,
                                    batch_size=embed_batch_size,
                                )

                            # precomputed가 약할 때 runtime_c1 retrieval로 폴백
                            if mode == "precomputed":
                                pre_pages_probe = provisional_pages_from_idxs(chunks, idxs, max_pages=5) if idxs else []
                                weak_precomputed = (not idxs) or (len(set(pre_pages_probe)) <= 1)
                                if weak_precomputed:
                                    rt_chunks, rt_artifact_path_str = cached_runtime_c1_chunks(doc_id, pdf_path, int(chunk_len))
                                    rt_texts = [c.text for c in rt_chunks]
                                    rt_sig = _artifact_signature(Path(rt_artifact_path_str))
                                    rt_source_key = f"runtime_c1__len{chunk_len}"

                                    rt_bm25 = None
                                    rt_faiss = None
                                    rt_embed = embed_model
                                    if retriever_mode == "R1":
                                        rt_bm25 = build_or_load_bm25(doc_id, rt_source_key, rt_sig, rt_texts)
                                        rt_idxs = retrieve_R1_bm25(rt_bm25, user_msg, retrieve_k)
                                    elif retriever_mode == "R2":
                                        rt_faiss, _ = build_or_load_faiss(
                                            doc_id, rt_source_key, rt_sig, rt_texts,
                                            embed_model_name, embed_batch_size, embed_device
                                        )
                                        if rt_embed is None:
                                            rt_embed = cached_embed_model(embed_model_name, embed_device)
                                        rt_idxs = retrieve_R2_flatip(rt_faiss, rt_embed, user_msg, retrieve_k, embed_batch_size)
                                    else:
                                        alpha = float(cfg_service.get("alpha", 0.7))
                                        bm25_candidates = int(cfg_service.get("bm25_candidates", 300))
                                        rt_bm25 = build_or_load_bm25(doc_id, rt_source_key, rt_sig, rt_texts)
                                        rt_faiss, _ = build_or_load_faiss(
                                            doc_id, rt_source_key, rt_sig, rt_texts,
                                            embed_model_name, embed_batch_size, embed_device
                                        )
                                        if rt_embed is None:
                                            rt_embed = cached_embed_model(embed_model_name, embed_device)
                                        rt_idxs = retrieve_R3_hybrid(
                                            rt_bm25, rt_faiss, rt_embed, user_msg,
                                            retrieve_k=retrieve_k,
                                            bm25_candidates=bm25_candidates,
                                            alpha=alpha,
                                            batch_size=embed_batch_size,
                                        )

                                    if rt_idxs:
                                        chunks = rt_chunks
                                        texts = rt_texts
                                        idxs = rt_idxs

                            if not idxs:
                                st.warning("검색 결과가 없습니다. 질문을 더 구체적으로 해보세요.")
                                st.session_state["svc_confirmed_pages"] = []
                                st.session_state["svc_candidate_pages"] = []
                                st.session_state["svc_highlight_queries"] = [user_msg[:20]]
                                try:
                                    n_pages2 = get_pdf_num_pages(pdf_path)
                                except Exception:
                                    n_pages2 = 1
                                sync_auto_navigate(get_start_page(doc_id), get_start_page(doc_id), n_pages2)
                                raise RuntimeError("Empty retrieval")

                            top_text = texts[int(idxs[0])]
                            timer["tick"]()

                            # 4) context
                            ctx_idxs = idxs[:context_k]
                            ctx_texts = [texts[i] for i in ctx_idxs if 0 <= i < len(texts)]
                            ev = "\n\n".join(ctx_texts)[: int(cfg_service.get("max_context_chars", 2500))]
                            prov_pages = provisional_pages_from_idxs(chunks, ctx_idxs, max_pages=3)

                            # 5) LLM
                            recent_history = _extract_recent_history(st.session_state.get("messages", []), max_msgs=6)

                            budget_won, budget_page, budget_line = _extract_budget_fact_from_ctx(chunks, ctx_idxs)
                            if budget_won > 0 and budget_page > 0:
                                st.session_state["svc_last_budget_fact"] = {
                                    "doc_id": doc_id,
                                    "won": int(budget_won),
                                    "page": int(budget_page),
                                    "line": str(budget_line),
                                }
                            else:
                                last_b = st.session_state.get("svc_last_budget_fact") or {}
                                if last_b.get("doc_id") == doc_id and int(last_b.get("won", 0)) > 0:
                                    budget_won = int(last_b.get("won", 0))
                                    budget_page = int(last_b.get("page", 0))
                                    budget_line = str(last_b.get("line", ""))

                            op, threshold_won = _extract_budget_threshold(user_msg)
                            local_judge_answer = ""
                            local_judge_pages: List[int] = []
                            if op and threshold_won > 0 and budget_won > 0:
                                verdict = False
                                if op == "<":
                                    verdict = budget_won < threshold_won
                                elif op == "<=":
                                    verdict = budget_won <= threshold_won
                                elif op == ">":
                                    verdict = budget_won > threshold_won
                                elif op == ">=":
                                    verdict = budget_won >= threshold_won
                                yn = "네" if verdict else "아니요"
                                local_judge_answer = (
                                    f"{yn}. 이 사업의 소요예산은 {_format_won(budget_won)}이며, "
                                    f"질문 기준({_format_won(threshold_won)} {op})과 비교하면 {'해당합니다' if verdict else '해당하지 않습니다'}."
                                )
                                if budget_page > 0:
                                    local_judge_pages = [budget_page]

                            if local_judge_answer:
                                raw_answer = local_judge_answer + (
                                    f"\n근거 페이지: p.{int(local_judge_pages[0])}" if local_judge_pages else "\n근거 페이지: NOT_FOUND"
                                )
                                if local_judge_pages:
                                    prov_pages = local_judge_pages + [p for p in prov_pages if p not in local_judge_pages]
                            else:
                                facts = []
                                if budget_won > 0:
                                    facts.append(f"확인된 예산: {_format_won(budget_won)}")
                                if budget_page > 0:
                                    facts.append(f"예산 근거 페이지 후보: p.{budget_page}")
                                memory_facts = " | ".join(facts)

                                raw_answer = summarize_with_evidence(
                                    api_key=api_key,
                                    model="gpt-5-mini",
                                    query=user_msg,
                                    evidence=ev,
                                    pages=prov_pages,
                                    chat_history=recent_history,
                                    memory_facts=memory_facts,
                                    temperature=float(cfg_service.get("temperature", 0.1)),
                                    max_completion_tokens=int(cfg_service.get("max_completion_tokens", 800)),
                                    fallback_model=str(cfg_service.get("fallback_model", "gpt-4.1-mini")),
                                    max_retries=int(cfg_service.get("generator_retries", 1)),
                                    reasoning_effort=str(cfg_service.get("reasoning_effort", "low")),
                                )

                            answer_clean = clean_answer_remove_page_lines(raw_answer)
                            st.markdown(answer_clean)

                            # 6) confirmed/candidate pages
                            try:
                                n_pages_route = get_pdf_num_pages(pdf_path)
                            except Exception:
                                n_pages_route = 1
                            start_page_doc = get_start_page(doc_id)
                            banned_pages_doc = get_banned_pages_for_routing(
                                doc_id=doc_id,
                                pdf_path=pdf_path,
                                start_page=start_page_doc,
                                n_pages=n_pages_route,
                            )

                            intent_now = detect_query_intent(user_msg)
                            if intent_now == "name":
                                name_pages = _name_answer_match_pages(
                                    pdf_path=pdf_path,
                                    answer=raw_answer,
                                    start_page=start_page_doc,
                                    n_pages=n_pages_route,
                                    banned_pages=set(banned_pages_doc),
                                )
                                if name_pages:
                                    prov_pages = _unique_pages_in_order(name_pages, prov_pages)

                            weak_q = [kw for kw in WEAK_KEYWORDS if kw in (user_msg + " " + raw_answer)]

                            if bool(cfg_service.get("enable_pdf_scan", True)):
                                strong_q_scan = extract_strong_queries_keywords(user_msg, raw_answer, top_text)
                                confirmed, candidates = pick_pages_confirmed_candidate(
                                    pdf_path,
                                    strong_q_scan,
                                    weak_q,
                                    cfg_service,
                                    fallback_pages=prov_pages,
                                    question=user_msg,
                                    banned_pages=banned_pages_doc,
                                    n_pages=n_pages_route,
                                    start_page=start_page_doc,
                                )
                            else:
                                confirmed = prov_pages[: int(cfg_service.get("confirmed_max", 2))]
                                candidates = []

                            st.session_state["svc_confirmed_pages"] = confirmed
                            st.session_state["svc_candidate_pages"] = candidates

                            banned_pages = set(banned_pages_doc)
                            page_try_order = _unique_pages_in_order(
                                [p for p in confirmed if p not in banned_pages],
                                [p for p in candidates if p not in banned_pages],
                                [p for p in prov_pages if p not in banned_pages],
                                [start_page_doc],
                            )
                            target = page_try_order[0] if page_try_order else start_page_doc

                            hl: List[str] = []
                            for p_try in page_try_order:
                                hl_try = build_best_highlights_for_page(
                                    pdf_path=pdf_path,
                                    page_phys=int(p_try),
                                    question=user_msg,
                                    answer=raw_answer,
                                    evidence=ev,
                                    cfg_obj=cfg_service,
                                )
                                if hl_try:
                                    target = int(p_try)
                                    hl = hl_try
                                    break

                            if (not hl) and bool(cfg_service.get("enable_hl_fallback_keywords", True)):
                                hl = extract_strong_queries_keywords(user_msg, raw_answer, top_text)
                                if not hl:
                                    hl = [t for t in extract_key_tokens(user_msg, raw_answer, cfg_service) if t][:1]

                            st.session_state["svc_highlight_queries"] = hl

                            try:
                                n_pages2 = get_pdf_num_pages(pdf_path)
                            except Exception:
                                n_pages2 = 1
                            sync_auto_navigate(int(target), start_page_doc, n_pages2)
                            # ====== ⬆⬆⬆ 여기까지는 네 기존 답변 생성 로직 "그대로" ======

                        finally:
                            # ✅ 기존 기능 유지: elapsed 저장/표시
                            elapsed_s = float(time.perf_counter() - float(timer["t0"]))
                            timer["stop"]()
                            st.caption(f"⏱ {elapsed_s:.2f}s")

                    # ✅ 기존 기능 유지: rerun 후에도 elapsed 보이도록 저장
                    st.session_state["messages"].append({"role": "assistant", "content": answer_clean, "elapsed_s": elapsed_s})

                    # ✅ 답변 생성 직후에도 맨 아래로 스크롤(체감 개선)
                    st.markdown(f'<div id="{CHAT_ANCHOR_ID}"></div>', unsafe_allow_html=True)
                    _autoscroll_to_bottom(CHAT_ANCHOR_ID)

                st.rerun()

# =========================================================
# Settings tab
# =========================================================
with tab_settings:
    st.subheader("⚙️ 서비스 설정")
    st.caption("저장하면 service_config.json과 fixed_config.py에 반영됩니다.")
    cfg_now = load_cfg()

    colA, colB = st.columns(2)

    with colA:
        service_chunk_mode = st.selectbox(
            "service_chunk_mode",
            ["precomputed", "runtime_c1"],
            index=0 if str(cfg_now.get("service_chunk_mode", "runtime_c1")) == "precomputed" else 1,
        )
        pp_version = st.selectbox(
            "pp_version (precomputed일 때)",
            pp_versions,
            index=pp_versions.index(cfg_now.get("pp_version", "pp_v5"))
            if str(cfg_now.get("pp_version", "pp_v5")) in pp_versions else 0,
        )
        chunk_length = st.slider("chunk_length", 300, 4000, int(cfg_now.get("chunk_length", 800)), 50)

        service_retriever_mode = st.selectbox(
            "service_retriever_mode",
            ["R1", "R2", "R3"],
            index=["R1", "R2", "R3"].index(str(cfg_now.get("service_retriever_mode", "R2")).upper())
            if str(cfg_now.get("service_retriever_mode", "R2")).upper() in ["R1", "R2", "R3"] else 1,
        )

        service_retrieve_k = st.slider("retrieve_k", 2, 200, int(cfg_now.get("service_retrieve_k", 20)), 1)
        service_context_k = st.slider("context_k", 2, 120, int(cfg_now.get("service_context_k", cfg_now.get("top_k", 16))), 1)

    with colB:
        alpha = st.slider("alpha (R3)", 0.0, 1.0, float(cfg_now.get("alpha", 0.7)), 0.05)
        bm25_candidates = st.slider("bm25_candidates (R3)", 50, 1000, int(cfg_now.get("bm25_candidates", 300)), 50)

        embed_model_name = st.text_input("embed_model_name (R2/R3)", value=str(cfg_now.get("embed_model_name", "nlpai-lab/KoE5")))
        embed_batch_size = st.slider("embed_batch_size (R2/R3)", 1, 256, int(cfg_now.get("embed_batch_size", 64)), 1)
        embed_device = st.selectbox(
            "embed_device (R2/R3)",
            ["cuda", "cpu"],
            index=0 if str(cfg_now.get("embed_device", "cuda")).lower() == "cuda" else 1,
        )

        max_context_chars = st.slider("max_context_chars", 500, 20000, int(cfg_now.get("max_context_chars", 2500)), 100)
        max_completion_tokens = st.slider("max_completion_tokens", 128, 4096, int(cfg_now.get("max_completion_tokens", 800)), 64)
        temperature = st.slider("temperature", 0.0, 1.5, float(cfg_now.get("temperature", 0.1)), 0.05)

    with st.expander("PDF 스캔 옵션", expanded=False):
        enable_pdf_scan = st.checkbox("enable_pdf_scan", value=bool(cfg_now.get("enable_pdf_scan", True)))
        max_pages_scan = st.slider("max_pages_scan", 10, 600, int(cfg_now.get("max_pages_scan", 160)), 10)
        confirmed_max = st.slider("confirmed_max", 1, 4, int(cfg_now.get("confirmed_max", 2)), 1)
        candidate_max = st.slider("candidate_max", 0, 30, int(cfg_now.get("candidate_max", 12)), 1)

    with st.expander("하이라이트 정확도 옵션", expanded=False):
        hl_window = st.slider("hl_window(페이지 substring 길이)", 25, 120, int(cfg_now.get("hl_window", 70)), 5)
        hl_max_phrases = st.slider("hl_max_phrases(하이라이트 개수)", 1, 12, int(cfg_now.get("hl_max_phrases", 8)), 1)
        hl_max_token = st.slider("hl_max_token(질문 토큰 수)", 3, 20, int(cfg_now.get("hl_max_token", 10)), 1)
        hl_min_score = st.slider("hl_min_score(토큰 동시 포함 점수)", 0, 5, int(cfg_now.get("hl_min_score", 1)), 1)
        enable_hl_fallback_keywords = st.checkbox(
            "enable_hl_fallback_keywords(정밀 실패 시 키워드 폴백)",
            value=bool(cfg_now.get("enable_hl_fallback_keywords", True)),
        )

    with st.expander("LLM 옵션", expanded=False):
        reasoning_effort = st.selectbox(
            "reasoning_effort",
            ["minimal", "low", "medium", "high"],
            index=["minimal", "low", "medium", "high"].index(str(cfg_now.get("reasoning_effort", "low")))
            if str(cfg_now.get("reasoning_effort", "low")) in ["minimal", "low", "medium", "high"] else 1,
        )
        generator_retries = st.slider("generator_retries", 0, 5, int(cfg_now.get("generator_retries", 1)), 1)
        fallback_model = st.text_input("fallback_model", value=str(cfg_now.get("fallback_model", "gpt-4.1-mini")))

    if st.button("✅ 저장(서비스 반영 + fixed_config.py 덮어쓰기)"):
        rtv = int(service_retrieve_k)
        ctx = int(service_context_k)
        if rtv < ctx:
            rtv = ctx

        cfg_new = {
            "service_chunk_mode": str(service_chunk_mode),
            "pp_version": str(pp_version),
            "chunk_length": int(chunk_length),

            "service_retriever_mode": str(service_retriever_mode),
            "service_retrieve_k": int(rtv),
            "service_context_k": int(ctx),

            "alpha": float(alpha),
            "bm25_candidates": int(bm25_candidates),

            "embed_model_name": str(embed_model_name),
            "embed_batch_size": int(embed_batch_size),
            "embed_device": str(embed_device),

            "max_context_chars": int(max_context_chars),
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),

            "enable_pdf_scan": bool(enable_pdf_scan),
            "max_pages_scan": int(max_pages_scan),
            "confirmed_max": int(confirmed_max),
            "candidate_max": int(candidate_max),

            "hl_window": int(hl_window),
            "hl_max_phrases": int(hl_max_phrases),
            "hl_max_token": int(hl_max_token),
            "hl_min_score": int(hl_min_score),
            "enable_hl_fallback_keywords": bool(enable_hl_fallback_keywords),

            "reasoning_effort": str(reasoning_effort),
            "generator_retries": int(generator_retries),
            "fallback_model": str(fallback_model),

            "top_k": int(ctx),
        }

        save_cfg(cfg_new)
        fixed_out = {**(FIXED_DEFAULT or {}), **cfg_new}
        fixed_out.setdefault("max_tokens", 2000)
        write_fixed_config_py(FIXED_CONFIG_PY, fixed_out)
        st.success("저장 완료! 서비스에 반영됩니다.")