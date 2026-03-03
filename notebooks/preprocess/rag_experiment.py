# preprocess/rag_experiment_chul.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gc
import json
import re  # ✅ [추가] 반복문자 제거용
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

try:
    from preprocess import pp_chul as pp
except ImportError:
    try:
        from preprocess import pp_v5 as pp
    except ImportError:  # v5/v6 미배포 전 백업
        from preprocess import pp_v4 as pp

ALL_DATA = pp.ALL_DATA
clean_text = pp.clean_text
extract_text = pp.extract_text
chunk_from_alldata = getattr(pp, "chunk_from_alldata")


def _chunk_for_index(doc_name: str, size: int) -> List[str] | None:
    try:
        return chunk_from_alldata(doc_name, size=size, include_meta=True)
    except TypeError:
        return chunk_from_alldata(doc_name, size=size)


# =========================================================
# PDF 겹침으로 생기는 반복문자(4회 이상) 축약
#   예: '2222222' -> '2', 'ㅋㅋㅋㅋ' -> 'ㅋ', '----' -> '-'
# =========================================================
_REPEAT_CHAR_4PLUS = re.compile(r"(.)\1{3,}")  # same char repeated >= 4

def squash_repeated_chars(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    return _REPEAT_CHAR_4PLUS.sub(r"\1", t)


# -------------------------
# Config / Prompt
# -------------------------
CONFIG = {
    "chunk_length": 800,
    "chunk_overlap": 200,
    "top_k": 9,                # backward compatibility
    "retrieve_k": 9,
    "context_k": 3,
    "recall_k": 5,
    "max_tokens": 2000,
    "max_completion_tokens": 2000,
    "temperature": 0.1,
    "alpha": 0.7,
    "bm25_candidates": 300,
    "max_context_chars": 6000,  # backward compatibility
    "max_context_chars_per_question": 3000,
    "batch_questions": 5,
    "reasoning_effort": "medium",
    "generator_retries": 1,
    "generator_fallback_single": True,
    "embed_batch_size": 64,
}


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(CONFIG.get(key, default))
    except Exception:
        return int(default)


def _resolve_k(
    top_k: Optional[int],
    retrieve_k: Optional[int],
    context_k: Optional[int],
    recall_k: Optional[int],
) -> Tuple[int, int, int]:
    rtv_k = int(retrieve_k) if retrieve_k is not None else (
        int(top_k) if top_k is not None else _cfg_int("retrieve_k", _cfg_int("top_k", 20))
    )
    ctx_k = int(context_k) if context_k is not None else _cfg_int("context_k", rtv_k)
    rcl_k = int(recall_k) if recall_k is not None else _cfg_int("recall_k", rtv_k)

    rtv_k = max(1, rtv_k)
    ctx_k = max(1, min(ctx_k, rtv_k))
    rcl_k = max(1, min(rcl_k, rtv_k))
    return rtv_k, ctx_k, rcl_k

RFP_PROMPT = """역할: 너는 RFP/입찰 공고 문서(CONTEXT 발췌)에서 정보를 추출한다.

절대 규칙:
1) 근거는 CONTEXT에 있는 문자열만 사용한다(추측 금지).
2) 출력은 JSON 객체 1개만. 코드블록/설명/추가 텍스트 금지.
3) 키는 QUESTIONS의 key를 정확히 그대로 사용한다(키 추가/삭제/변경 금지).
4) 값은 모두 string으로 출력한다.
5) CONTEXT에 명확한 근거가 있을 때만 채워라. 애매하거나 유사한 것 같아도 확신할 수 없으면 NOT_FOUND를 써라.
6) "NOT_FOUND"는 정말로 근거가 전혀 없을 때만 사용한다.
7) 날짜는 문서에 나온 형식 그대로 사용한다. 조사/어미 붙이지 말 것. (예: "~까지", "~부터" 금지)
8) 금액은 문서에 나온 형식 그대로 사용한다. 단위(원)가 문서에 있으면 붙이고, 없으면 붙이지 말 것. (예: "100,000,000원" 또는 "100,000,000")
9) 값은 최대한 짧고 핵심만. 단, 아래 12~14번 규칙이 있는 필드는 해당 규칙을 우선 적용하라.
10) 라벨/항목명은 값에 포함하지 말 것. (예: "발주기관 : 국민연금공단" → "국민연금공단", "사업기간 : 6개월" → "6개월")
11) 금액 표기 시 괄호 안 부연설명 붙이지 말 것. 숫자와 단위만. "부가세 포함", "VAT포함", "원정", "금 ~만원" 등 모두 제거. (예: "50,000,000(금 오천만원/VAT포함)" → "50,000,000원", "금243,000,000원(VAT포함)" → "243,000,000원")
12) requirements_must, eligibility는 문서에 나열된 항목을 " / "로 구분해서 나열하라. 임의로 요약하거나 생략하지 말 것. (예: "SW사업자 등록 / 최근 3년 실적 1건 이상 / 정보보안관리체계 인증")
13) eval_items는 "항목명:배점" 형식으로 " / "로 구분해서 나열하라. (예: "기술평가:90 / 가격평가:10")
14) contract_type은 문서에 나온 계약 방식 명칭을 그대로 나열하라.
    여러 개면 " / "로 구분. (예: "제한경쟁입찰 / 협상에 의한 계약")
15) agency(발주기관)는 기관명만 출력하라. 직위/직책은 붙이지 말 것. (예: "부산국제영화제 집행위원장" → "부산국제영화제", "국립민속박물관장" → "국립민속박물관")

작업 방법(반드시 따름):
- 먼저 CONTEXT에서 다음 유형의 신호를 찾아라: 사업명/용역명, 금액(원), 기간(일/개월), 기관명, 마감일, 평가(기술/가격), 요구사항/자격/평가항목/계약방식/사업목적.
- 찾은 신호가 있으면 해당 key에 매핑해 값을 채워라.
- 확실한 매핑이 불가능하면 NOT_FOUND.

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
    """Baseline chunk: fixed chars with optional overlap"""
    def __init__(self, size: int = 800, overlap: int = 0):
        self.size = size
        self.overlap = max(0, int(overlap))

    def chunk(self, doc_path: Path) -> List[str]:
        text = clean_text(extract_text(doc_path))
        text = squash_repeated_chars(text)  # ✅ [추가] 노이즈 제거
        s = self.size
        if s <= 0:
            return []
        step = max(1, s - min(self.overlap, max(0, s - 1)))
        return [text[i:i+s] for i in range(0, len(text), step)]


class C2PageChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        chunks: List[str] = []
        with pdfplumber.open(doc_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = clean_text(page.extract_text() or "")
                page_text = squash_repeated_chars(page_text)  # ✅ [추가] 노이즈 제거
                if page_text:
                    chunks.append(f"[페이지 {i+1}]\n{page_text}")
        return chunks


class C3SectionChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        chunks = _chunk_for_index(doc_path.name, size=CONFIG["chunk_length"])
        if chunks is not None:
            return [squash_repeated_chars(c) for c in chunks]  # ✅ [추가] 노이즈 제거
        text = clean_text(extract_text(doc_path))
        text = squash_repeated_chars(text)  # ✅ [추가] 노이즈 제거
        s = CONFIG["chunk_length"]
        return [text[i:i+s] for i in range(0, len(text), s)]


class C4DoclingChunker(BaseChunker):
    def chunk(self, doc_path: Path) -> List[str]:
        chunks = _chunk_for_index(doc_path.name, size=CONFIG["chunk_length"])
        if chunks is not None:
            return [squash_repeated_chars(c) for c in chunks]  # ✅ [추가] 노이즈 제거
        return C1FixedChunker(
            size=CONFIG["chunk_length"],
            overlap=_cfg_int("chunk_overlap", 0),
        ).chunk(doc_path)


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
    """Baseline vector: KoE5 embeddings + FAISS IndexFlatL2"""
    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model

    def build_index(self, chunks: List[str]) -> Any:
        embs = self.embed_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs.astype("float32"))
        return index

    def retrieve(self, index: Any, query_texts: List[str], top_k: int) -> List[int]:
        q_embs = self.embed_model.encode(
            query_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
        q_mean = q_embs.mean(axis=0, keepdims=True)
        _, I = index.search(q_mean.astype("float32"), top_k)
        return [int(i) for i in I[0]]


class R3HybridRetriever(BaseRetriever):
    """Hybrid: BM25 + Vector"""
    def __init__(self, embed_model: SentenceTransformer, bm25_candidates: int = 200):
        self.embed_model = embed_model
        self.bm25_candidates = bm25_candidates

    def build_index(self, chunks: List[str]) -> Any:
        bm25 = rank_bm25.BM25Okapi([c.split() for c in chunks])
        embs = self.embed_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
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

        q_embs = self.embed_model.encode(
            query_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
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


class R4RerankerRetriever(BaseRetriever):
    def __init__(self, embed_model, reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.embed_model = embed_model
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(reranker_model, use_fp16=True)

    def build_index(self, chunks: List[str]) -> Any:
        embs = self.embed_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs.astype("float32"))
        return index

    def retrieve(self, index: Any, queries: List[str], top_k: int = 20) -> List[int]:
        q_embs = self.embed_model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=_cfg_int("embed_batch_size", 64),
        )
        q_mean = q_embs.mean(axis=0, keepdims=True).astype("float32")
        _, I = index.search(q_mean, min(top_k * 3, index.ntotal))
        candidates = I[0].tolist()
        return candidates[:top_k]


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

        MAX_CTX_CHARS = _cfg_int(
            "max_context_chars_per_question",
            _cfg_int("max_context_chars", 6000),
        )
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

        retries = max(1, _cfg_int("generator_retries", 1))
        reasoning_effort = str(CONFIG.get("reasoning_effort", "minimal"))

        for _ in range(retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=CONFIG.get("max_completion_tokens", 2000),
                    reasoning={"effort": reasoning_effort},
                )
                self.last_resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else None
                self.last_debug["response_status"] = getattr(resp, "status", None)

                usage = getattr(resp, "usage", None)
                self.last_debug["output_tokens"] = getattr(usage, "output_tokens", None)

                text = (getattr(resp, "output_text", "") or "").strip()
                self.last_raw_text = text
                self.last_debug["output_text_repr"] = repr(text[:200])

                if not text:
                    continue

                try:
                    obj = json.loads(text)
                except Exception as e:
                    self.last_debug["parse_error"] = repr(e)
                    continue

                if not isinstance(obj, dict):
                    self.last_debug["parse_error"] = f"non-dict-json: {type(obj)}"
                    continue

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

        if bool(CONFIG.get("generator_fallback_single", True)) and len(queries) > 1:
            out: Dict[str, str] = {}
            for k, q in queries:
                out.update(self.generate([(k, q)], context))
            return out

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
        chunker = C1FixedChunker(
            size=CONFIG["chunk_length"],
            overlap=_cfg_int("chunk_overlap", 0),
        )
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
        retriever = R3HybridRetriever(
            embed_model,
            bm25_candidates=_cfg_int("bm25_candidates", 200),
        )
    elif spec.retriever == "R4":
        retriever = R4RerankerRetriever(embed_model)
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
        top_k: Optional[int] = None,  # backward compatibility
        retrieve_k: Optional[int] = None,
        context_k: Optional[int] = None,
        recall_k: Optional[int] = None,
        sim_threshold: int = 80,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFKC", doc_path.name)
        doc_name = re.sub(r'\s+', ' ', doc_name).strip()

        queries = get_queries_for_doc(doc_name, self.questions_df)
        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)

        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()
        GOLD_ANCHOR = build_gold_anchor_map(gold_evidence_df)

        pred_map: Dict[str, str] = {}
        g_list: List[Dict[str, float]] = []
        r_list: List[Dict[str, float]] = []
        rtv_k, ctx_k, rcl_k = _resolve_k(top_k, retrieve_k, context_k, recall_k)

        for field, question in queries:
            idxs = self.retriever.retrieve(index, [question], top_k=rtv_k)
            ctx_idxs = idxs[:ctx_k]
            rcl_idxs = idxs[:rcl_k]
            context = "".join(chunks[int(i)] for i in ctx_idxs if 0 <= int(i) < len(chunks))

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
                    r_list.append(eval_retrieval_by_anchor(chunks, rcl_idxs, anchors))
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
        top_k: Optional[int] = None,  # backward compatibility
        retrieve_k: Optional[int] = None,
        context_k: Optional[int] = None,
        recall_k: Optional[int] = None,
        sim_threshold: int = 80,
        warn_on_mismatch: bool = True,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFKC", doc_path.name)
        doc_name = re.sub(r'\s+', ' ', doc_name).strip()

        queries = get_queries_for_doc(doc_name, self.questions_df)
        q_texts = [q for _t, q in queries]
        type_keys = [t for t, _q in queries]
        rtv_k, ctx_k, rcl_k = _resolve_k(top_k, retrieve_k, context_k, recall_k)

        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)
        idxs = self.retriever.retrieve(index, q_texts, top_k=rtv_k)
        ctx_idxs = idxs[:ctx_k]
        rcl_idxs = idxs[:rcl_k]

        context = "".join(chunks[int(i)] for i in ctx_idxs if 0 <= int(i) < len(chunks))
        batch_size = max(1, _cfg_int("batch_questions", len(queries) or 1))
        if batch_size >= len(queries):
            pred_map = self.generator.generate(queries, context)
        else:
            pred_map = {}
            for i in range(0, len(queries), batch_size):
                pred_map.update(self.generator.generate(queries[i:i+batch_size], context))

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
                r_list.append(eval_retrieval_by_anchor(chunks, rcl_idxs, anchors))
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
