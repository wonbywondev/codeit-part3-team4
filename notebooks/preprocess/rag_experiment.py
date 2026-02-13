# preprocess/rag_experiment.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gc
import json
import unicodedata

import numpy as np
import pandas as pd

import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import rank_bm25
from rapidfuzz import fuzz

from preprocess.pp_basic import EVAL_DIR

# ✅ ImportError만 fallback (pp_v5가 "없을 때"만 pp_v4 사용)
try:
    from preprocess import pp_v5 as pp
except ImportError as e:  # v5 배포 전/백업
    print("[WARN] pp_v5 import 실패(ImportError), pp_v4로 fallback:", repr(e))
    from preprocess import pp_v4 as pp

ALL_DATA = getattr(pp, "ALL_DATA", None)
clean_text = pp.clean_text
extract_text = pp.extract_text

chunk_from_alldata = getattr(pp, "chunk_from_alldata", None)


# -------------------------
# Config / Prompt
# -------------------------
CONFIG = {
    "chunk_length": 1200,          # C1 baseline
    "top_k": 20,
    "max_tokens": 2000,            # non gpt-5 (현재 코드에서는 미사용)
    "max_completion_tokens": 2000, # gpt-5
    "temperature": 0.1,            # non gpt-5 (현재 코드에서는 미사용)
    "alpha": 0.7,                  # hybrid weight for vector score
    "max_context_chars": 6000,     # context hard cap (chars)
}

RFP_PROMPT = """역할: 너는 RFP/입찰 공고 문서(CONTEXT 발췌)에서 정보를 추출한다.

절대 규칙:
1) 근거는 CONTEXT에 있는 문자열만 사용한다(추측 금지).
2) 출력은 JSON 객체 1개만. 코드블록/설명/추가 텍스트 금지.
3) 키는 QUESTIONS의 key를 정확히 그대로 사용한다(키 추가/삭제/변경 금지).
4) 값은 모두 string으로 출력한다.
5) CONTEXT에 관련 단서(부분일치, 유사표현, 숫자, 기관명 후보)가 1개라도 있으면 가능한 범위에서 채워라.
6) "NOT_FOUND"는 정말로 근거가 전혀 없을 때만 사용한다.
7) 모든 값을 "NOT_FOUND"로 채우는 출력은 금지한다. (최소 1개는 CONTEXT에서 발췌해 채워라)

QUESTIONS(JSON array):
{questions_json}

CONTEXT:
{context}
""".strip()


# -------------------------
# Baseline-compatible utils
# -------------------------
def load_questions_df() -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / "questions.csv")


def get_queries_for_doc(doc_name: str, questions_df: pd.DataFrame) -> List[Tuple[str, str]]:
    common = questions_df[questions_df["doc_id"] == "*"][["type", "question"]]
    per_doc = questions_df[questions_df["doc_id"] == doc_name][["type", "question"]]
    merged = pd.concat([common, per_doc], ignore_index=True)

    merged["type"] = merged["type"].astype(str)
    merged["question"] = merged["question"].astype(str)

    merged = merged.drop_duplicates(subset=["type"], keep="last")
    return list(zip(merged["type"].tolist(), merged["question"].tolist()))


def eval_retrieval_by_anchor(chunks: List[str], idxs: List[int], anchors: List[str]) -> Dict[str, float]:
    hit_rank = None
    for rank, ci in enumerate(idxs, start=1):
        if 0 <= int(ci) < len(chunks):
            c = chunks[int(ci)]
            if any(a in c for a in anchors):
                hit_rank = rank
                break
    return {"recall": 1.0 if hit_rank else 0.0, "mrr": (1.0 / hit_rank) if hit_rank else 0.0}


def eval_gen(pred: str, gold: Optional[str], threshold: int = 80) -> Dict[str, float]:
    pred = (pred or "").strip()
    fill = 1.0 if pred and pred.lower() not in {"", "없음"} else 0.0

    if gold is None or str(gold).strip() == "":
        return {"fill": fill, "match": np.nan, "sim": np.nan}

    gold = str(gold).strip()
    sim = fuzz.token_set_ratio(pred, gold)
    return {"fill": fill, "match": 1.0 if sim >= threshold else 0.0, "sim": float(sim)}


def build_gold_anchor_map(gold_evidence_df: pd.DataFrame) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for _, r in gold_evidence_df.iterrows():
        iid = str(r["instance_id"])
        anchor = str(r.get("anchor_text", "") or "").strip()
        if anchor:
            m.setdefault(iid, []).append(anchor)
    return m


# -------------------------
# ABCs
# -------------------------
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc_path: Path) -> List[str]:
        ...


class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, chunks: List[str]) -> Any:
        ...

    @abstractmethod
    def retrieve(self, index: Any, query_texts: List[str], top_k: int) -> List[int]:
        ...


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, queries: List[Tuple[str, str]], context: str) -> Dict[str, str]:
        ...
        

# -------------------------
# Chunkers
# -------------------------
class C1FixedChunker(BaseChunker):
    def __init__(self, size: int = 800):
        self.size = size

    def chunk(self, doc_path: Path) -> List[str]:
        text = clean_text(extract_text(doc_path))
        s = self.size
        return [text[i:i+s] for i in range(0, len(text), s)]


class C2PageChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        chunks: List[str] = []
        with pdfplumber.open(doc_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = clean_text(page.extract_text() or "")
                if page_text:
                    chunks.append(f"[페이지 {i+1}]\n{page_text}")
        return chunks


class C3SectionChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        if callable(chunk_from_alldata):
            chunks = chunk_from_alldata(doc_path.name, size=CONFIG["chunk_length"])
            if chunks is not None:
                return chunks

        text = clean_text(extract_text(doc_path))
        s = CONFIG["chunk_length"]
        return [text[i:i+s] for i in range(0, len(text), s)]


class C4DoclingChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        if callable(chunk_from_alldata):
            chunks = chunk_from_alldata(doc_path.name, size=CONFIG["chunk_length"])
            if chunks is not None:
                return chunks
        return C1FixedChunker(size=CONFIG["chunk_length"]).chunk(doc_path)


# -------------------------
# Retrievers
# -------------------------
class R1BM25Retriever(BaseRetriever):
    def build_index(self, chunks: List[str]) -> Any:
        tokenized = [c.split() for c in chunks]
        return rank_bm25.BM25Okapi(tokenized)

    def retrieve(self, bm25_index: Any, query_texts: List[str], top_k: int) -> List[int]:
        q = " ".join(query_texts).split()
        scores = bm25_index.get_scores(q)
        top = np.argsort(scores)[::-1][:top_k]
        return top.astype(int).tolist()


class R2VectorRetriever(BaseRetriever):
    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model

    def build_index(self, chunks: List[str]) -> Any:
        embs = self.embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs.astype("float32"))
        return index

    def retrieve(self, index: Any, query_texts: List[str], top_k: int) -> List[int]:
        q_embs = self.embed_model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False)
        q_mean = q_embs.mean(axis=0, keepdims=True)
        _, I = index.search(q_mean.astype("float32"), top_k)
        return [int(i) for i in I[0]]


class R3HybridRetriever(BaseRetriever):
    def __init__(self, embed_model: SentenceTransformer, bm25_candidates: int = 200):
        self.embed_model = embed_model
        self.bm25_candidates = bm25_candidates

    def build_index(self, chunks: List[str]) -> Any:
        bm25 = rank_bm25.BM25Okapi([c.split() for c in chunks])
        embs = self.embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss_index = faiss.IndexFlatL2(embs.shape[1])
        faiss_index.add(embs.astype("float32"))
        return {"bm25": bm25, "faiss": faiss_index, "chunks": chunks, "bm25_embs": embs}

    def retrieve(self, index: Any, query_texts: List[str], top_k: int) -> List[int]:
        bm25 = index["bm25"]
        faiss_index = index["faiss"]
        chunks = index["chunks"]

        q_text = " ".join(query_texts)

        bm25_scores = bm25.get_scores(q_text.split())
        cand_n = min(self.bm25_candidates, len(chunks))
        cand_idxs = np.argsort(bm25_scores)[::-1][:cand_n].astype(int)

        q_embs = self.embed_model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False)
        q_mean = q_embs.mean(axis=0, keepdims=True).astype("float32")

        _, vec_I = faiss_index.search(q_mean, min(max(top_k, cand_n), len(chunks)))
        vec_idxs = vec_I[0].astype(int)

        union = np.unique(np.concatenate([cand_idxs, vec_idxs]))

        bm = bm25_scores[union]

        vec_rank_score = np.zeros(len(chunks), dtype=np.float32)
        for rank, idx in enumerate(vec_idxs, start=1):
            vec_rank_score[idx] = 1.0 / rank

        vv = vec_rank_score[union]
        hybrid = CONFIG["alpha"] * vv + (1.0 - CONFIG["alpha"]) * bm

        top = union[np.argsort(hybrid)[::-1][:top_k]]
        return top.astype(int).tolist()


# -------------------------
# Generators
# -------------------------
class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
        self.model = model

        self.last_raw_text: str = ""
        self.last_resp_dump: Optional[Dict[str, Any]] = None
        self.last_debug: Dict[str, Any] = {}

    def generate(self, queries: List[Tuple[str, str]], context: str) -> Dict[str, str]:
        NOT_FOUND = "NOT_FOUND"
        GEN_FAIL = "GEN_FAIL"

        MAX_CTX_CHARS = CONFIG.get("max_context_chars", 6000)
        context = (context or "")[:MAX_CTX_CHARS]

        q_payload = [{"key": k, "question": q} for k, q in queries]
        questions_json = json.dumps(q_payload, ensure_ascii=False)
        prompt = RFP_PROMPT.format(questions_json=questions_json, context=context)

        self.last_raw_text = ""
        self.last_resp_dump = None
        self.last_debug = {
            "model": self.model,
            "n_questions": len(queries),
            "context_len": len(context or ""),
            "max_context_chars": MAX_CTX_CHARS,
            "prompt_len": len(prompt),
            "response_status": None,
            "output_tokens": None,
            "output_text_repr": None,
            "exception": None,
            "parse_error": None,
        }

        def all_sentinel(s: str) -> Dict[str, str]:
            return {k: s for k, _ in queries}

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=CONFIG.get("max_completion_tokens", 2000),
                reasoning={"effort": "minimal"},
            )
            self.last_resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else None
            self.last_debug["response_status"] = getattr(resp, "status", None)

            usage = getattr(resp, "usage", None)
            self.last_debug["output_tokens"] = getattr(usage, "output_tokens", None)

            text = (getattr(resp, "output_text", "") or "").strip()
            self.last_raw_text = text
            self.last_debug["output_text_repr"] = repr(text[:200])

            if not text:
                return all_sentinel(GEN_FAIL)

            try:
                obj = json.loads(text)
            except Exception as e:
                self.last_debug["parse_error"] = repr(e)
                return all_sentinel(GEN_FAIL)

            if not isinstance(obj, dict):
                self.last_debug["parse_error"] = f"non-dict-json: {type(obj)}"
                return all_sentinel(GEN_FAIL)

            out: Dict[str, str] = {}
            for k, _q in queries:
                if k not in obj:
                    out[k] = NOT_FOUND
                    continue

                v_raw = obj.get(k)
                v = (v_raw or "").strip()
                out[k] = v if v else GEN_FAIL

            return out

        except Exception as e:
            self.last_debug["exception"] = repr(e)
            self.last_raw_text = ""
            return all_sentinel(GEN_FAIL)


# -------------------------
# Experiment runner
# -------------------------
@dataclass
class ExperimentSpec:
    exp_id: int
    chunker: str
    retriever: str
    generator: str


def make_components(spec: ExperimentSpec, embed_model: SentenceTransformer, client: OpenAI):
    if spec.chunker == "C1":
        chunker = C1FixedChunker(size=CONFIG["chunk_length"])
    elif spec.chunker == "C2":
        chunker = C2PageChunker()
    elif spec.chunker == "C3":
        chunker = C3SectionChunker()
    elif spec.chunker == "C4":
        chunker = C4DoclingChunker()
    else:
        raise ValueError(spec.chunker)

    if spec.retriever == "R1":
        retriever = R1BM25Retriever()
    elif spec.retriever == "R2":
        retriever = R2VectorRetriever(embed_model)
    elif spec.retriever == "R3":
        retriever = R3HybridRetriever(embed_model, bm25_candidates=200)
    else:
        raise ValueError(spec.retriever)

    if spec.generator == "G1":
        gen = OpenAIGenerator(model="gpt-5-mini", client=client)
    elif spec.generator == "G2":
        gen = OpenAIGenerator(model="gpt-5-nano", client=client)
    else:
        raise ValueError(spec.generator)

    return chunker, retriever, gen


class RAGExperiment:
    def __init__(self, chunker: BaseChunker, retriever: BaseRetriever, generator: BaseGenerator, questions_df: pd.DataFrame):
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self.questions_df = questions_df

    def run_single_doc_metrics_singleq(
        self,
        doc_path: Path,
        gold_fields_df: pd.DataFrame,
        gold_evidence_df: pd.DataFrame,
        top_k: int = 20,
        sim_threshold: int = 80,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFC", doc_path.name)

        queries = get_queries_for_doc(doc_name, self.questions_df)
        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)

        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()
        GOLD_ANCHOR = build_gold_anchor_map(gold_evidence_df)

        pred_map: Dict[str, str] = {}
        g_list: List[Dict[str, float]] = []
        r_list: List[Dict[str, float]] = []

        for field, question in queries:
            idxs = self.retriever.retrieve(index, [question], top_k=top_k)
            context = "".join(chunks[int(i)] for i in idxs if 0 <= int(i) < len(chunks))

            one_pred = self.generator.generate([(field, question)], context)
            pred = (one_pred.get(field) or "").strip()
            pred_map[field] = pred

            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None
            g_list.append(eval_gen(pred, gold, threshold=sim_threshold))

            for _, row in qdf[qdf["field"].astype(str) == str(field)].iterrows():
                iid = str(row["instance_id"])
                anchors = GOLD_ANCHOR.get(iid, [])
                if anchors:
                    r_list.append(eval_retrieval_by_anchor(chunks, idxs, anchors))
                else:
                    r_list.append({"recall": np.nan, "mrr": np.nan})

        metrics: Dict[str, Any] = {
            "doc_id": doc_name,
            "n_questions": int(len(qdf)),
            "chunk_count": int(len(chunks)),
            "pred_map": pred_map,

            "ret_recall": float(np.nanmean([x["recall"] for x in r_list])) if r_list else np.nan,
            "ret_mrr": float(np.nanmean([x["mrr"] for x in r_list])) if r_list else np.nan,

            "gen_fill": float(np.nanmean([x["fill"] for x in g_list])) if g_list else np.nan,
            "gen_match": float(np.nanmean([x["match"] for x in g_list])) if g_list else np.nan,
            "gen_sim": float(np.nanmean([x["sim"] for x in g_list])) if g_list else np.nan,
        }

        del chunks, index, qdf, r_list, g_list, queries, GOLD_ANCHOR, pred_map
        gc.collect()
        return metrics

    def run_single_doc_metrics(
        self,
        doc_path: Path,
        gold_fields_df: pd.DataFrame,
        gold_evidence_df: pd.DataFrame,
        top_k: int = 20,
        sim_threshold: int = 80,
        warn_on_mismatch: bool = True,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFC", doc_path.name)

        queries = get_queries_for_doc(doc_name, self.questions_df)
        q_texts = [q for _t, q in queries]
        type_keys = [t for t, _q in queries]

        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)
        idxs = self.retriever.retrieve(index, q_texts, top_k=top_k)

        context = "".join(chunks[int(i)] for i in idxs if 0 <= int(i) < len(chunks))
        pred_map = self.generator.generate(queries, context)

        answers = [pred_map.get(t, "NOT_FOUND") for t in type_keys]

        expected_answer_count = len(q_texts)
        answer_count = len(answers)
        if warn_on_mismatch and answer_count != expected_answer_count:
            print(
                f"WARN answer_count mismatch | doc={doc_name} | "
                f"expected={expected_answer_count} got={answer_count}"
            )

        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()
        GOLD_ANCHOR = build_gold_anchor_map(gold_evidence_df)

        answers_preview = [str(x) for x in (answers[:5] if answers else [])]
        n_nonempty_answers = int(sum(1 for a in (answers or []) if str(a).strip()))
        n_notfound_answers = int(sum(1 for a in (answers or []) if str(a).strip().lower() in {"notfound", "not_found", "없음"}))

        raw_text = getattr(self.generator, "last_raw_text", None)
        raw_text_len = None if raw_text is None else int(len(str(raw_text).strip()))
        raw_text_preview = None if raw_text is None else str(raw_text)[:200]

        g_list: List[Dict[str, float]] = []
        preds: List[str] = []

        for i, (field, _q) in enumerate(queries):
            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None

            pred = answers[i] if i < len(answers) else ""
            pred_s = (pred or "").strip()
            preds.append(pred_s)

            g = eval_gen(pred_s, gold, threshold=sim_threshold)
            g_list.append(g)

        pred_preview = preds[:5]
        n_nonempty_preds = int(sum(1 for p in preds if str(p).strip()))
        n_notfound_preds = int(sum(1 for p in preds if str(p).strip().lower() in {"notfound", "not_found", "없음"}))

        r_list: List[Dict[str, float]] = []
        for _, row in qdf.iterrows():
            iid = str(row["instance_id"])
            anchors = GOLD_ANCHOR.get(iid, [])
            if anchors:
                r_list.append(eval_retrieval_by_anchor(chunks, idxs, anchors))
            else:
                r_list.append({"recall": np.nan, "mrr": np.nan})

        metrics: Dict[str, Any] = {
            "doc_id": doc_name,
            "expected_answer_count": int(expected_answer_count),
            "answer_count": int(answer_count),

            "n_questions": int(len(qdf)),
            "chunk_count": int(len(chunks)),
            "context_length": int(len(context)),

            "raw_text_len": raw_text_len,
            "raw_text_preview": raw_text_preview,
            "answers_preview": answers_preview,
            "n_nonempty_answers": n_nonempty_answers,
            "n_notfound_answers": n_notfound_answers,
            "pred_preview": pred_preview,
            "n_nonempty_preds": n_nonempty_preds,
            "n_notfound_preds": n_notfound_preds,

            "pred_map": pred_map,

            "ret_recall": float(np.nanmean([x["recall"] for x in r_list])),
            "ret_mrr": float(np.nanmean([x["mrr"] for x in r_list])),

            "gen_fill": float(np.nanmean([x["fill"] for x in g_list])),
            "gen_match": float(np.nanmean([x["match"] for x in g_list])),
            "gen_sim": float(np.nanmean([x["sim"] for x in g_list])),
        }

        del chunks, index, context, answers, qdf, r_list, g_list, idxs, queries, q_texts, GOLD_ANCHOR, preds, pred_map
        gc.collect()
        return metrics