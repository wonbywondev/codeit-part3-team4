# notebooks/preprocess/ragas_eval.py
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 같은 CONFIG/spec/component을 그대로 재사용
from preprocess.rag_experiment import (
    CONFIG,
    ExperimentSpec,
    load_questions_df,
    make_components,
    get_queries_for_doc,
)

# ------------------------------------------------------------
# Gold loader (fields.jsonl -> long df)
# ------------------------------------------------------------
def load_gold_fields_jsonl(path: str | Path) -> pd.DataFrame:
    """
    gold_fields.jsonl 형태:
      {"instance_id": "...", "doc_id": "...", "fields": {"fieldA": "goldA", ...}}
    -> long df:
      instance_id, doc_id, field, gold
    """
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    out = []
    for r in rows:
        iid = r.get("instance_id")
        doc_id = r.get("doc_id", "")
        fields = r.get("fields", {}) or {}
        for k, v in fields.items():
            out.append(
                {"instance_id": iid, "doc_id": doc_id, "field": str(k), "gold": v}
            )
    return pd.DataFrame(out)


def _gold_map_for_doc(gold_fields_df: pd.DataFrame, doc_name: str) -> Dict[str, str]:
    qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == str(doc_name)].copy()
    m: Dict[str, str] = {}
    for _, r in qdf.iterrows():
        field = str(r["field"])
        gold = r.get("gold", None)
        if gold is None:
            continue
        gold_s = str(gold).strip()
        if gold_s:
            m[field] = gold_s
    return m


# ------------------------------------------------------------
# Build RAGAS-style rows from the SAME run (same CONFIG)
# ------------------------------------------------------------
def build_ragas_rows_for_doc(
    doc_path: Path,
    questions_df: pd.DataFrame,
    gold_fields_df: pd.DataFrame,
    chunker,
    retriever,
    generator,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      ragas_rows: [{user_input, response, retrieved_contexts, reference, doc_id, field}, ...]
      doc_meta: debug meta per doc
    """
    doc_name = unicodedata.normalize("NFC", doc_path.name)
    queries: List[Tuple[str, str]] = get_queries_for_doc(doc_name, questions_df)
    q_texts = [q for _t, q in queries]

    chunks: List[str] = chunker.chunk(doc_path)
    index = retriever.build_index(chunks)
    idxs: List[int] = retriever.retrieve(index, q_texts, top_k=top_k)

    # contexts는 list[str] 유지 (RAGAS 입력)
    contexts: List[str] = [chunks[int(i)] for i in idxs if 0 <= int(i) < len(chunks)]

    # generator는 내부에서 CONFIG["max_context_chars"]로 자름 (동일 조건 보장)
    pred_map: Dict[str, str] = generator.generate(queries, "".join(contexts))

    gold_map = _gold_map_for_doc(gold_fields_df, doc_name)

    rows: List[Dict[str, Any]] = []
    for field, question in queries:
        rows.append(
            {
                "user_input": str(question),
                "response": (pred_map.get(field, "") or "").strip(),
                "retrieved_contexts": contexts,  # list[str]
                "reference": gold_map.get(str(field), None),  # str|None
                "doc_id": doc_name,
                "field": str(field),
            }
        )

    doc_meta = {
        "doc_id": doc_name,
        "chunk_count": int(len(chunks)),
        "top_k": int(top_k),
        "contexts_count": int(len(contexts)),
        "contexts_joined_len": int(sum(len(c) for c in contexts)),
        "n_questions": int(len(queries)),
        "max_context_chars": int(CONFIG.get("max_context_chars", 0)),
    }
    return rows, doc_meta


# ------------------------------------------------------------
# GPT-5 evaluator (Responses API) - no temperature
# ------------------------------------------------------------
JUDGE_PROMPT = """너는 RAG 시스템 답변을 평가하는 엄격한 평가자다.
아래 입력을 보고, JSON 형식으로만 평가 결과를 출력하라. 설명/코드블록/추가 텍스트 금지.

지표 정의 (0~1 점수):
- faithfulness: 답변의 핵심 주장들이 retrieved_contexts에서 직접 확인되는 정도
- context_precision: retrieved_contexts가 질문/답에 유용한 정보 위주인지(노이즈 적을수록 높음)
- answer_correctness: reference(정답)가 있을 때 답변이 맞는 정도. reference가 없으면 null.

평가 규칙:
- retrieved_contexts에 없는 내용을 답변이 주장하면 faithfulness는 낮아야 한다.
- 컨텍스트가 길고 무관한 내용이 많으면 context_precision은 낮아야 한다.
- 답변이 NOT_FOUND/GEN_FAIL 또는 공백이면 세 지표 모두 낮게 평가하라.
- answer_correctness는 reference가 비어있으면 반드시 null로 출력하라.

출력 JSON 스키마(키 고정):
{{
  "faithfulness": <number 0..1>,
  "context_precision": <number 0..1>,
  "answer_correctness": <number 0..1 or null>
}}

INPUT:
user_input: {user_input}
response: {response}
reference: {reference}
retrieved_contexts:
{retrieved_contexts}
"""


def _clip01(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    except Exception:
        return None


def run_ragas_gpt5(
    rows: List[Dict[str, Any]],
    client: OpenAI,
    evaluator_model: str = "gpt-5-mini",
    max_context_chars_per_sample: int = 6000,
    max_output_tokens: int = 500,
    reasoning_effort: str = "minimal",
) -> pd.DataFrame:
    """
    GPT-5 Responses API로 RAGAS 스타일 지표(3개)를 계산.
    - temperature 파라미터를 사용하지 않음 (GPT-5 제약 회피)
    """
    out_rows: List[Dict[str, Any]] = []

    for r in tqdm(rows, desc="GPT-5 judge scoring"):
        user_input = str(r.get("user_input", ""))
        response = str(r.get("response", ""))
        reference = r.get("reference", None)
        ref_s = "" if reference is None else str(reference)

        ctx_list = r.get("retrieved_contexts", []) or []
        ctx_joined = "\n\n".join([str(x) for x in ctx_list])
        ctx_joined = ctx_joined[:max_context_chars_per_sample]

        prompt = JUDGE_PROMPT.format(
            user_input=user_input,
            response=response,
            reference=ref_s,
            retrieved_contexts=ctx_joined,
        )

        faith = None
        cprec = None
        acorr = None
        raw_text = ""

        try:
            resp = client.responses.create(
                model=evaluator_model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": reasoning_effort},
            )
            raw_text = (getattr(resp, "output_text", "") or "").strip()

            obj = json.loads(raw_text)
            faith = _clip01(obj.get("faithfulness"))
            cprec = _clip01(obj.get("context_precision"))
            acorr = obj.get("answer_correctness")
            acorr = None if acorr is None else _clip01(acorr)

        except Exception:
            # 실패하면 NaN(None)로 둠 (전체 실행이 죽지 않게)
            pass

        out_rows.append(
            {
                "exp_id": r.get("exp_id"),
                "chunker": r.get("chunker"),
                "retriever": r.get("retriever"),
                "generator": r.get("generator"),
                "doc_id": r.get("doc_id"),
                "field": r.get("field"),
                "user_input": r.get("user_input"),
                "faithfulness": faith,
                "context_precision": cprec,
                "answer_correctness": acorr,
                # 디버그가 필요하면 아래를 저장해도 됨(용량 커질 수 있음)
                # "judge_raw": raw_text[:500],
            }
        )

    return pd.DataFrame(out_rows)


# ------------------------------------------------------------
# Result container
# ------------------------------------------------------------
@dataclass
class RagasRunResult:
    # 기존(ret/gen) 문서 단위 DF
    doc_metrics_df: pd.DataFrame
    # 질문 단위 RAGAS 점수 DF (doc_id/field 단위)
    ragas_sample_df: pd.DataFrame
    # 문서 단위 평균 RAGAS DF
    ragas_doc_df: pd.DataFrame
    # exp 단위 평균 RAGAS DF
    ragas_exp_df: pd.DataFrame


# ------------------------------------------------------------
# Main entry for exp_notebook (one-call)
# ------------------------------------------------------------
def run_experiment_with_ragas(
    spec: ExperimentSpec,
    run_docs: List[str | Path],
    gold_fields_jsonl_path: str | Path,
    embed_model: SentenceTransformer,
    client: OpenAI,
    evaluator_model: str = "gpt-5-mini",
    ragas_metrics: Optional[List[str]] = None,  # 현재는 3개만 지원
    compute_baseline_doc_metrics: bool = True,
    gold_evidence_df: Optional[pd.DataFrame] = None,
    sim_threshold: int = 80,
    # judge 옵션
    judge_max_context_chars_per_sample: int = 6000,
    judge_max_output_tokens: int = 500,
    judge_reasoning_effort: str = "minimal",
) -> RagasRunResult:
    """
    exp_notebook에서 한 번에 호출:
    - 같은 spec/run_docs/CONFIG로 RAG 실행
    - (옵션) 기존 ret/gen 문서 지표 계산
    - 같은 실행 결과로 RAGAS 스타일 지표(3개)를 GPT-5로 평가

    주의:
    - ragas_metrics는 현재 ["faithfulness","context_precision","answer_correctness"]만 지원(temperature 이슈 회피 목적)
    """
    if ragas_metrics is None:
        ragas_metrics = ["faithfulness", "context_precision", "answer_correctness"]

    allowed = {"faithfulness", "context_precision", "answer_correctness"}
    bad = [m for m in ragas_metrics if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported metrics {bad}. Allowed: {sorted(allowed)}")

    questions_df = load_questions_df()
    gold_fields_df = load_gold_fields_jsonl(gold_fields_jsonl_path)

    chunker, retriever, generator = make_components(spec, embed_model=embed_model, client=client)
    top_k = int(CONFIG.get("top_k", 15))

    # 1) build ragas rows for all docs
    all_rows: List[Dict[str, Any]] = []
    for dp in tqdm([Path(p) for p in run_docs], desc=f"RAG + RAGAS | exp {spec.exp_id}"):
        rows, _meta = build_ragas_rows_for_doc(
            doc_path=Path(dp),
            questions_df=questions_df,
            gold_fields_df=gold_fields_df,
            chunker=chunker,
            retriever=retriever,
            generator=generator,
            top_k=top_k,
        )
        for r in rows:
            r["exp_id"] = spec.exp_id
            r["chunker"] = spec.chunker
            r["retriever"] = spec.retriever
            r["generator"] = spec.generator
        all_rows.extend(rows)

    # 2) judge (GPT-5)
    ragas_sample_df = run_ragas_gpt5(
        rows=all_rows,
        client=client,
        evaluator_model=evaluator_model,
        max_context_chars_per_sample=judge_max_context_chars_per_sample,
        max_output_tokens=judge_max_output_tokens,
        reasoning_effort=judge_reasoning_effort,
    )

    # (필요한 metric만 남기기)
    keep_cols = ["exp_id","chunker","retriever","generator","doc_id","field","user_input"] + ragas_metrics
    ragas_sample_df = ragas_sample_df[keep_cols].copy()

    # 3) doc-level avg
    ragas_doc_df = ragas_sample_df.groupby("doc_id")[ragas_metrics].mean(numeric_only=True).reset_index()

    # 4) exp-level avg
    exp_avg = ragas_sample_df[ragas_metrics].mean(numeric_only=True)
    ragas_exp_df = pd.DataFrame([{"exp_id": spec.exp_id, **{k: float(exp_avg[k]) for k in exp_avg.index}}])

    # 5) baseline doc metrics (optional)
    if compute_baseline_doc_metrics:
        if gold_evidence_df is None:
            raise ValueError("compute_baseline_doc_metrics=True면 gold_evidence_df를 넘겨줘야 함")

        from preprocess.rag_experiment import RAGExperiment
        rag = RAGExperiment(chunker=chunker, retriever=retriever, generator=generator, questions_df=questions_df)

        doc_rows = []
        for dp in tqdm([Path(p) for p in run_docs], desc=f"Baseline doc metrics | exp {spec.exp_id}"):
            m = rag.run_single_doc_metrics(
                Path(dp),
                gold_fields_df=gold_fields_df,
                gold_evidence_df=gold_evidence_df,
                top_k=top_k,
                sim_threshold=sim_threshold,
            )
            m["exp_id"] = spec.exp_id
            m["chunker"] = spec.chunker
            m["retriever"] = spec.retriever
            m["generator"] = spec.generator
            doc_rows.append(m)

        doc_metrics_df = pd.DataFrame(doc_rows)
    else:
        doc_metrics_df = pd.DataFrame([])

    return RagasRunResult(
        doc_metrics_df=doc_metrics_df,
        ragas_sample_df=ragas_sample_df,
        ragas_doc_df=ragas_doc_df,
        ragas_exp_df=ragas_exp_df,
    )