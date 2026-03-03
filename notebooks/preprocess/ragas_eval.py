# notebooks/preprocess/ragas_eval.py
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from preprocess.rag_experiment import (
    CONFIG,
    ExperimentSpec,
    load_questions_df,
    make_components,
    get_queries_for_doc,
)

def load_gold_fields_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out = []
    for r in rows:
        iid = r.get("instance_id")
        doc_id = r.get("doc_id", "")
        fields = r.get("fields", {}) or {}
        for k, v in fields.items():
            out.append({"instance_id": iid, "doc_id": doc_id, "field": str(k), "gold": v})
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

def _dedupe_ints_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs or []:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out

def build_ragas_rows_for_doc(
    doc_path: Path,
    questions_df: pd.DataFrame,
    gold_fields_df: pd.DataFrame,
    chunker,
    retriever,
    generator,
    *,
    retrieve_k: int,
    context_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc_name = unicodedata.normalize("NFC", doc_path.name)
    queries: List[Tuple[str, str]] = get_queries_for_doc(doc_name, questions_df)

    if not queries:
        return [], {"doc_id": doc_name, "n_questions": 0, "skipped": "no_queries"}

    chunks: List[str] = chunker.chunk(doc_path)
    index = retriever.build_index(chunks)
    gold_map = _gold_map_for_doc(gold_fields_df, doc_name)

    rows: List[Dict[str, Any]] = []
    contexts_count_list: List[int] = []
    contexts_joined_len_list: List[int] = []
    for field, question in queries:
        idxs: List[int] = retriever.retrieve(index, [question], top_k=int(retrieve_k))
        idxs = _dedupe_ints_keep_order(idxs)
        ctx_idxs = idxs[: int(context_k)]
        contexts: List[str] = [chunks[int(i)] for i in ctx_idxs if 0 <= int(i) < len(chunks)]
        one_pred = generator.generate([(field, question)], "".join(contexts))

        contexts_count_list.append(int(len(contexts)))
        contexts_joined_len_list.append(int(sum(len(c) for c in contexts)))
        rows.append(
            {
                "user_input": str(question),
                "response": (one_pred.get(field, "") or "").strip(),
                "retrieved_contexts": contexts,
                "reference": gold_map.get(str(field), None),
                "doc_id": doc_name,
                "field": str(field),
            }
        )

    doc_meta = {
        "doc_id": doc_name,
        "chunk_count": int(len(chunks)),
        "retrieve_k": int(retrieve_k),
        "context_k": int(context_k),
        "contexts_count_avg": float(sum(contexts_count_list) / len(contexts_count_list)) if contexts_count_list else 0.0,
        "contexts_joined_len_avg": float(sum(contexts_joined_len_list) / len(contexts_joined_len_list)) if contexts_joined_len_list else 0.0,
        "n_questions": int(len(queries)),
        "max_context_chars": int(CONFIG.get("max_context_chars", 0)),
    }
    return rows, doc_meta


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
        if v != v:
            return None
        return max(0.0, min(1.0, v))
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
    expected_cols = [
        "exp_id", "chunker", "retriever", "generator",
        "doc_id", "field", "user_input",
        "faithfulness", "context_precision", "answer_correctness",
    ]
    if not rows:
        return pd.DataFrame(columns=expected_cols)

    out_rows: List[Dict[str, Any]] = []
    for r in tqdm(rows, desc="GPT-5 judge scoring"):
        user_input = str(r.get("user_input", ""))
        response = str(r.get("response", ""))
        reference = r.get("reference", None)
        ref_s = "" if reference is None else str(reference)

        ctx_list = r.get("retrieved_contexts", []) or []
        ctx_joined = "\n\n".join([str(x) for x in ctx_list])[:max_context_chars_per_sample]

        prompt = (
            JUDGE_PROMPT
            .replace("{user_input}", user_input)
            .replace("{response}", response)
            .replace("{reference}", ref_s)
            .replace("{retrieved_contexts}", ctx_joined)
        )

        faith = None
        cprec = None
        acorr = None
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
            ac = obj.get("answer_correctness")
            acorr = None if ac is None else _clip01(ac)
        except Exception:
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
            }
        )

    df = pd.DataFrame(out_rows)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].copy()


@dataclass
class RagasRunResult:
    doc_metrics_df: pd.DataFrame
    ragas_sample_df: pd.DataFrame
    ragas_doc_df: pd.DataFrame
    ragas_exp_df: pd.DataFrame


def run_experiment_with_ragas(
    spec: ExperimentSpec,
    run_docs: List[str | Path],
    gold_fields_jsonl_path: str | Path,
    embed_model: SentenceTransformer,
    client: OpenAI,
    evaluator_model: str = "gpt-5-mini",
    ragas_metrics: Optional[List[str]] = None,
    compute_baseline_doc_metrics: bool = True,
    gold_evidence_df: Optional[pd.DataFrame] = None,
    sim_threshold: int = 80,
    # ✅ 3-K: RAGAS는 retrieve_k / context_k만 받으면 됨(지표는 없으니까)
    retrieve_k: Optional[int] = None,
    context_k: Optional[int] = None,
    # judge 옵션
    judge_max_context_chars_per_sample: int = 6000,
    judge_max_output_tokens: int = 500,
    judge_reasoning_effort: str = "minimal",
) -> RagasRunResult:
    if ragas_metrics is None:
        ragas_metrics = ["faithfulness", "context_precision", "answer_correctness"]

    questions_df = load_questions_df()
    gold_fields_df = load_gold_fields_jsonl(gold_fields_jsonl_path)

    chunker, retriever, generator = make_components(spec, embed_model=embed_model, client=client)

    ctx_k = int(context_k) if context_k is not None else 8
    rtv_k = int(retrieve_k) if retrieve_k is not None else max(ctx_k, int(CONFIG.get("top_k", 15)))
    if rtv_k < ctx_k:
        rtv_k = ctx_k

    all_rows: List[Dict[str, Any]] = []
    for dp in tqdm([Path(p) for p in run_docs], desc=f"RAG + RAGAS | exp {spec.exp_id}"):
        rows, _meta = build_ragas_rows_for_doc(
            doc_path=Path(dp),
            questions_df=questions_df,
            gold_fields_df=gold_fields_df,
            chunker=chunker,
            retriever=retriever,
            generator=generator,
            retrieve_k=rtv_k,
            context_k=ctx_k,
        )
        for r in rows:
            r["exp_id"] = spec.exp_id
            r["chunker"] = spec.chunker
            r["retriever"] = spec.retriever
            r["generator"] = spec.generator
        all_rows.extend(rows)

    ragas_sample_df = run_ragas_gpt5(
        rows=all_rows,
        client=client,
        evaluator_model=evaluator_model,
        max_context_chars_per_sample=judge_max_context_chars_per_sample,
        max_output_tokens=judge_max_output_tokens,
        reasoning_effort=judge_reasoning_effort,
    )

    keep_cols = ["exp_id","chunker","retriever","generator","doc_id","field","user_input"] + ragas_metrics
    for c in keep_cols:
        if c not in ragas_sample_df.columns:
            ragas_sample_df[c] = None
    ragas_sample_df = ragas_sample_df[keep_cols].copy()

    if len(ragas_sample_df) == 0:
        ragas_doc_df = pd.DataFrame(columns=["doc_id"] + ragas_metrics)
        ragas_exp_df = pd.DataFrame([{"exp_id": spec.exp_id, **{m: float("nan") for m in ragas_metrics}}])
    else:
        ragas_doc_df = ragas_sample_df.groupby("doc_id")[ragas_metrics].mean(numeric_only=True).reset_index()
        exp_avg = ragas_sample_df[ragas_metrics].mean(numeric_only=True)
        ragas_exp_df = pd.DataFrame([{"exp_id": spec.exp_id, **{k: float(exp_avg[k]) for k in exp_avg.index}}])

    if compute_baseline_doc_metrics:
        if gold_evidence_df is None:
            raise ValueError("compute_baseline_doc_metrics=True면 gold_evidence_df 필요")

        from preprocess.rag_experiment import RAGExperiment
        rag = RAGExperiment(chunker=chunker, retriever=retriever, generator=generator, questions_df=questions_df)

        doc_rows = []
        for dp in tqdm([Path(p) for p in run_docs], desc=f"Baseline doc metrics | exp {spec.exp_id}"):
            m = rag.run_single_doc_metrics(
                Path(dp),
                gold_fields_df=gold_fields_df,
                gold_evidence_df=gold_evidence_df,
                retrieve_k=rtv_k,
                context_k=ctx_k,
                recall_k=max(ctx_k, int(CONFIG.get("top_k", 15))),  # fallback
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
