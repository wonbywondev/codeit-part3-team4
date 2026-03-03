# streamlit_app/app.py
from __future__ import annotations

import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd

from core.config import AppPaths, AppConfig
from core.config_io import write_fixed_config_py
from core.loaders import make_pdf_map, get_chunks
from core.preprocess_runner import detect_available_pp_versions, run_preprocessing
from core.retriever import build_or_load_hybrid, evidence_text
from core.llm import summarize_with_evidence
from core.render import (
    get_pdf_num_pages,
    render_pdf_page_png,
    render_pdf_page_png_with_highlights,
    find_pages_with_hit_counts,
    find_pages_for_queries,
)

# ----------------------------
# init
# ----------------------------
paths = AppPaths()
cfg = AppConfig()

CACHE_DIR = paths.cache_dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_CFG_PATH = CACHE_DIR / "service_config.json"
FIXED_CONFIG_PY = paths.repo_root / "streamlit_app" / "core" / "fixed_config.py"

DEFAULT_CFG = {
    "pp_version": "pp_v5",
    "chunk_length": 1200,
    "top_k": 8,
    "alpha": 0.7,
    "max_context_chars": 2500,
    "max_completion_tokens": 800,
    "temperature": 0.1,
    "confirmed_max": 2,
    "candidate_max": 12,
    "max_pages_scan": 200,
}

WEAK_KEYWORDS = ["예산", "사업비", "소요예산", "부가가치세", "VAT", "계약", "입찰", "기간", "마감", "제출", "기관", "요구사항"]

# ----------------------------
# small caches
# ----------------------------
@st.cache_resource(show_spinner=False)
def cached_import(module_name: str):
    import importlib
    return importlib.import_module(module_name)

@st.cache_resource(show_spinner=False)
def cached_embed_model(model_name: str = "BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

# =========================================================
# Config IO
# =========================================================
def load_cfg() -> Dict[str, Any]:
    if SERVICE_CFG_PATH.exists():
        try:
            data = json.loads(SERVICE_CFG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **data}
        except Exception:
            return DEFAULT_CFG.copy()
    return DEFAULT_CFG.copy()

def save_cfg(cfg_obj: Dict[str, Any]) -> None:
    SERVICE_CFG_PATH.write_text(json.dumps(cfg_obj, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================================================
# 01_index_pages.json 로드 (목차 페이지 스킵용)
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
            label = "표지"
        elif p in index_labels:
            label = "목차"
        else:
            label = "앞부분"
        result.append((p, label))
    return result

# =========================================================
# Chat session-only (restart -> reset, doc switch -> keep)
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

# =========================================================
# Retrieval helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def provisional_pages_from_results(results: list, max_pages: int = 3) -> List[int]:
    pages = []
    for r in (results or [])[:5]:
        try:
            p = int(getattr(r.chunk, "page", 0) or 0)
        except Exception:
            p = 0
        if p > 0 and p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break
    return pages

def extract_strong_queries(question: str, answer: str, top_chunk_text: str) -> List[str]:
    q = _norm(question)
    a = _norm(answer)
    lines = [x.strip() for x in (top_chunk_text or "").replace("\r","\n").split("\n") if x.strip()]
    out: List[str] = []

    quoted = re.findall(r'"([^"]+)"', a)
    for s in quoted[:2]:
        out.append(_norm(s)[:80])

    if ":" in a:
        out.append(_norm(a.split(":")[-1])[:80])

    for kw in WEAK_KEYWORDS:
        if kw in q:
            for ln in lines:
                if kw in ln and len(ln) >= 8:
                    out.append(_norm(ln)[:80])
                    break
            break

    if any(k in q for k in ["예산", "사업비", "소요예산", "금액"]):
        nums = re.findall(r"\b\d[\d,]{4,}\b", a)
        if nums:
            n = nums[0]
            out.append(n)
            out.append(n.replace(",", ""))

    if not out and a:
        out.append(a[:60])

    uniq = []
    seen = set()
    for x in out:
        x = _norm(x)
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:3]

def pick_pages_confirmed_candidate(
    pdf_path: str,
    strong_queries: List[str],
    weak_queries: List[str],
    cfg_obj: Dict[str, Any],
    fallback_pages: List[int],
    start_page: int = 1,
) -> Tuple[List[int], List[int]]:
    confirmed_max = int(cfg_obj["confirmed_max"])
    candidate_max = int(cfg_obj["candidate_max"])
    max_pages_scan = int(cfg_obj["max_pages_scan"])

    confirmed: List[int] = []
    candidate: List[int] = []

    hit_map = {}
    if strong_queries:
        try:
            hit_map = find_pages_with_hit_counts(pdf_path, strong_queries, start_page=start_page, max_pages_scan=max_pages_scan)
        except Exception:
            hit_map = {}

    if hit_map:
        ranked = sorted(hit_map.items(), key=lambda kv: (-kv[1], kv[0]))
        confirmed = [p for p, _ in ranked[:confirmed_max]]
        candidate = [p for p, _ in ranked[confirmed_max:confirmed_max + candidate_max]]
    else:
        confirmed = (fallback_pages or [])[:confirmed_max]
        if weak_queries:
            try:
                candidate = find_pages_for_queries(pdf_path, weak_queries, start_page=start_page, max_pages_scan=max_pages_scan)
            except Exception:
                candidate = []

    candidate = [p for p in candidate if p not in confirmed][:candidate_max]
    return confirmed, candidate

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="RAG Chat Service", layout="wide")
st.title("🤖 RAG 서비스 / 🧪 실험·평가")

pdf_map = make_pdf_map(paths.pdf_dir)
pdf_names = list(pdf_map.keys())
pp_versions = detect_available_pp_versions(paths) or ["pp_v5", "pp_v6", "pp_v4"]

tab_service, tab_exp_eval = st.tabs(["🟢 서비스(채팅)", "🧪 실험/평가(설정/평가)"])

# =========================================================
# 🟢 서비스(채팅)
# =========================================================
with tab_service:
    cfg_service = load_cfg()

    col_chat, col_view = st.columns([1.15, 0.85], gap="large")

    with col_view:
        st.subheader("📄 PDF / 근거 보기")
        doc_id = st.selectbox("문서 선택", pdf_names, key="svc_doc_select")
        pdf_path = str(pdf_map[doc_id])

        switch_doc(doc_id)

        confirmed_pages = st.session_state.get("svc_confirmed_pages", [])
        candidate_pages = st.session_state.get("svc_candidate_pages", [])
        highlight_queries = st.session_state.get("svc_highlight_queries", [])

        start_page = get_start_page(doc_id)
        front_pages = get_front_pages(doc_id)
        offset = start_page - 1

        def to_content(pages: List[int]) -> List[int]:
            return [p - offset for p in pages]

        st.markdown("**✅ 확정 근거 페이지(하이라이트 가능)**")
        st.write(to_content(confirmed_pages) if confirmed_pages else [])

        st.markdown("**🟡 후보 페이지(하이라이트 없음)**")
        st.write(to_content(candidate_pages) if candidate_pages else [])

        try:
            n_pages = get_pdf_num_pages(pdf_path)
        except Exception:
            n_pages = 1

        show_hl = st.checkbox("하이라이트 보기(확정 페이지에서만)", value=True, disabled=(not confirmed_pages))

        # ✅ 앞부분 버튼
        if front_pages:
            st.caption("앞부분 바로 보기")
            btn_cols = st.columns(len(front_pages))
            for col, (phys_p, lbl) in zip(btn_cols, front_pages):
                if col.button(lbl, key=f"svc_front_{phys_p}"):
                    st.session_state["svc_render_phys"] = phys_p

        content_max = max(1, n_pages - offset)
        base_pages = confirmed_pages if confirmed_pages else candidate_pages

        cur_phys = st.session_state.get("svc_render_phys")
        if cur_phys is not None and start_page <= cur_phys <= n_pages:
            default_content_p = cur_phys - offset
        elif base_pages:
            default_content_p = max(1, base_pages[0] - offset)
        else:
            default_content_p = 1
        default_content_p = max(1, min(default_content_p, content_max))

        st.number_input(
            f"본문 페이지 (1..{content_max})",
            min_value=1,
            max_value=content_max,
            value=default_content_p,
            step=1,
            key="svc_pv_page",
        )

        # ✅ 페이지 보기 버튼 추가
        if st.button("페이지 보기", key="svc_render_btn"):
            st.session_state["svc_render_phys"] = int(st.session_state["svc_pv_page"]) + offset

        page_to_view = st.session_state.get("svc_render_phys")
        if page_to_view is None:
            page_to_view = int(st.session_state.get("svc_pv_page", default_content_p)) + offset

        is_confirmed = page_to_view in (confirmed_pages or [])
        is_front = page_to_view < start_page
        if is_front:
            front_label = next((lbl for p, lbl in front_pages if p == page_to_view), "앞부분")
            caption = f"{front_label} (물리적 p.{page_to_view})"
        else:
            caption = f"p.{page_to_view - offset}"

        try:
            if show_hl and is_confirmed and highlight_queries:
                img, hits = render_pdf_page_png_with_highlights(pdf_path, int(page_to_view), highlight_queries, zoom=2.0)
                st.image(img, caption=f"{caption} (highlights={hits})", use_container_width=True)
            else:
                img = render_pdf_page_png(pdf_path, int(page_to_view), zoom=2.0)
                st.image(img, caption=caption, use_container_width=True)
        except Exception as e:
            st.error("페이지 렌더링 실패")
            st.exception(e)

    with col_chat:
        st.subheader("💬 채팅")

        for m in st.session_state.get("messages", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("질문을 입력하세요 (예: 사업명/예산/계약방식/기간/마감/기관/요구사항)")

        if user_msg:
            api_key = cfg.openai_api_key
            if not api_key:
                st.error("OPENAI_API_KEY가 없습니다. .env 확인 필요")
                st.stop()

            st.session_state["messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("검색/답변 생성 중..."):
                    service_pp = str(cfg_service.get("pp_version", "pp_v5"))

                    jsonl = paths.chunks_dir / service_pp / f"{doc_id}.jsonl"
                    if not jsonl.exists():
                        ok, msg = run_preprocessing(service_pp, doc_id, int(cfg_service["chunk_length"]), paths)
                        if not ok:
                            st.warning("전처리 생성 실패 → pdf_fallback으로 진행")
                            jsonl = None

                    source = "precomputed_chunks" if jsonl and jsonl.exists() else "pdf_fallback"
                    chunks, artifact_path = get_chunks(
                        doc_id=doc_id,
                        pdf_path=pdf_path,
                        paths=paths,
                        source=source,
                        precomputed_jsonl=(jsonl if source == "precomputed_chunks" else None),
                    )

                    retriever = build_or_load_hybrid(
                        chunks=chunks,
                        index_dir=paths.index_dir,
                        doc_id=doc_id,
                        source=f"{source}__{service_pp}",
                        artifact_path=artifact_path,
                    )

                    results = retriever.search(user_msg, k=int(cfg_service["top_k"]), alpha=float(cfg_service["alpha"]))
                    ev = evidence_text(results, max_chars=int(cfg_service["max_context_chars"]))
                    prov_pages = provisional_pages_from_results(results, max_pages=3)

                    answer = summarize_with_evidence(
                        api_key=api_key,
                        model="gpt-5-mini",
                        query=user_msg,
                        evidence=ev,
                        pages=prov_pages,
                        temperature=float(cfg_service["temperature"]),
                        max_completion_tokens=int(cfg_service["max_completion_tokens"]),
                    )
                    st.markdown(answer)

                    top_chunk_text = results[0].chunk.text if results else ""
                    strong_q = extract_strong_queries(user_msg, answer, top_chunk_text)
                    weak_q = [kw for kw in WEAK_KEYWORDS if kw in (user_msg + " " + answer)]

                    confirmed, candidates = pick_pages_confirmed_candidate(
                        pdf_path,
                        strong_q,
                        weak_q,
                        cfg_service,
                        fallback_pages=prov_pages,
                        start_page=get_start_page(doc_id),
                    )

                    st.session_state["svc_confirmed_pages"] = confirmed
                    st.session_state["svc_candidate_pages"] = candidates
                    st.session_state["svc_highlight_queries"] = strong_q if confirmed else []

                    st.session_state["messages"].append({"role": "assistant", "content": answer})


# =========================================================
# 🧪 실험/평가(설정 변경 + 평가 실행)
# =========================================================
with tab_exp_eval:
    st.subheader("🧪 실험/평가 설정")
    st.caption("여기서 저장한 값이 서비스에 자동 반영되며, fixed_config.py도 자동으로 덮어씁니다.")

    cfg_now = load_cfg()

    colA, colB = st.columns(2)
    with colA:
        pp_version = st.selectbox(
            "pp version(서비스에도 적용)",
            pp_versions,
            index=pp_versions.index(cfg_now["pp_version"]) if cfg_now["pp_version"] in pp_versions else 0
        )
        chunk_length = st.slider("chunk_length", 300, 2000, int(cfg_now["chunk_length"]), 100)
        top_k = st.slider("top_k", 2, 30, int(cfg_now["top_k"]), 1)
        alpha = st.slider("alpha(hybrid)", 0.0, 1.0, float(cfg_now["alpha"]), 0.05)
        max_context_chars = st.slider("max_context_chars", 500, 8000, int(cfg_now["max_context_chars"]), 100)

    with colB:
        max_completion_tokens = st.slider("max_completion_tokens", 128, 4096, int(cfg_now["max_completion_tokens"]), 64)
        temperature = st.slider("temperature", 0.0, 1.5, float(cfg_now["temperature"]), 0.05)
        confirmed_max = st.slider("confirmed_max", 1, 4, int(cfg_now["confirmed_max"]), 1)
        candidate_max = st.slider("candidate_max", 5, 30, int(cfg_now["candidate_max"]), 1)
        max_pages_scan = st.slider("max_pages_scan", 10, 400, int(cfg_now["max_pages_scan"]), 10)

    if st.button("✅ 저장 → 서비스 반영 + fixed_config.py 덮어쓰기"):
        cfg_new = {
            "pp_version": str(pp_version),
            "chunk_length": int(chunk_length),
            "top_k": int(top_k),
            "alpha": float(alpha),
            "max_context_chars": int(max_context_chars),
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "confirmed_max": int(confirmed_max),
            "candidate_max": int(candidate_max),
            "max_pages_scan": int(max_pages_scan),
        }
        save_cfg(cfg_new)

        fixed_out = {
            "pp_version": str(pp_version),
            "chunk_length": int(chunk_length),
            "top_k": int(top_k),
            "max_tokens": 2000,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
            "alpha": float(alpha),
            "max_context_chars": int(max_context_chars),
        }
        write_fixed_config_py(FIXED_CONFIG_PY, fixed_out)
        st.success("저장 완료! 서비스 반영 + fixed_config.py 업데이트 완료")

    st.divider()
    st.subheader("📊 평가 실행 (rag_experiment.py / ragas_eval.py 기반)")

    eval_chunker = st.selectbox("chunker", ["C1", "C2", "C3", "C4"], index=0, key="eval_chunker")
    eval_retriever = st.selectbox("retriever", ["R1", "R2", "R3"], index=2, key="eval_retriever")
    eval_generator = st.selectbox("generator", ["G1", "G2"], index=0, key="eval_generator")
    eval_top_k = st.slider("eval top_k", 2, 30, int(top_k), 1, key="eval_topk")
    eval_sim_threshold = st.slider("eval sim_threshold", 50, 100, 80, 1, key="eval_sim")
    run_ragas = st.checkbox("RAGAS 실행", value=True, key="eval_ragas")
    judge_model = st.selectbox("RAGAS Judge", ["gpt-5-mini", "gpt-5-nano"], index=0, key="eval_judge")

    if st.button("📊 평가 실행", key="eval_run_btn"):
        api_key = cfg.openai_api_key
        if not api_key:
            st.error("OPENAI_API_KEY가 비어있어요.")
            st.stop()

        with st.spinner("평가 실행 중..."):
            rag_experiment = cached_import("preprocess.rag_experiment")
            ragas_eval = cached_import("preprocess.ragas_eval")

            if hasattr(rag_experiment, "CONFIG") and isinstance(rag_experiment.CONFIG, dict):
                rag_experiment.CONFIG["chunk_length"] = int(chunk_length)

            questions_df = rag_experiment.load_questions_df()
            gold_evidence_df = pd.read_csv(paths.eval_dir / "gold_evidence.csv")
            gold_fields_df = ragas_eval.load_gold_fields_jsonl(paths.eval_dir / "gold_fields.jsonl")

            doc_names = sorted(set(questions_df.loc[questions_df["doc_id"] != "*", "doc_id"].astype(str).tolist()))
            run_docs = [paths.pdf_dir / name for name in doc_names if (paths.pdf_dir / name).exists()]

            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            embed_model = cached_embed_model("BAAI/bge-m3")

            spec = rag_experiment.ExperimentSpec(exp_id=0, chunker=eval_chunker, retriever=eval_retriever, generator=eval_generator)
            chunker_obj, retriever_obj, gen_obj = rag_experiment.make_components(spec, embed_model=embed_model, client=client)
            exp = rag_experiment.RAGExperiment(chunker_obj, retriever_obj, gen_obj, questions_df)

            rows = []
            for dp in run_docs:
                m = exp.run_single_doc_metrics(
                    doc_path=Path(dp),
                    gold_fields_df=gold_fields_df,
                    gold_evidence_df=gold_evidence_df,
                    top_k=int(eval_top_k),
                    sim_threshold=int(eval_sim_threshold),
                )
                rows.append(m)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            if run_ragas:
                ragas_res = ragas_eval.run_experiment_with_ragas(
                    spec=spec,
                    run_docs=run_docs,
                    gold_fields_jsonl_path=paths.eval_dir / "gold_fields.jsonl",
                    embed_model=embed_model,
                    client=client,
                    evaluator_model=judge_model,
                    ragas_metrics=["faithfulness", "context_precision", "answer_correctness"],
                    compute_baseline_doc_metrics=False,
                    gold_evidence_df=gold_evidence_df,
                    sim_threshold=int(eval_sim_threshold),
                )
                st.dataframe(ragas_res.ragas_doc_df, use_container_width=True)