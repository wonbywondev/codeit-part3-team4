# preprocess/rag_experiment.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gc
import numpy as np
import pandas as pd
import unicodedata

import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import rank_bm25
from rapidfuzz import fuzz

from preprocess.pp_basic import EVAL_DIR
from preprocess.pp_v4 import ALL_DATA, clean_text, extract_text, chunk_from_alldata


# -------------------------
# Config / Prompt
# -------------------------
CONFIG = {
    "chunk_length": 800,     # C1 baseline
    "top_k": 15,
    "max_tokens": 2000,
    "max_completion_tokens": 2000,
    "temperature": 0.1,
    "alpha": 0.7,            # hybrid weight for vector score
}

RFP_PROMPT = """
너는 정부·공공기관 제안요청서(RFP)를 분석하는 전문가다.
아래 컨텍스트는 하나의 정부 RFP 문서에서 추출된 내용이다.

[분석 규칙]
- 추측 금지, 문서에 명시된 내용만 사용
- 문서에 없으면 반드시 NOT_FOUND
- 질문 개수만큼 답변을 반드시 모두 출력
- 각 답변은 반드시 지정된 태그로 감싸서 출력
- 태그 밖에는 어떤 문자(설명/번호/공백/개행 포함)도 출력하지 말 것

[출력 형식(필수)]
- i번째 질문의 답변은 반드시 정확히 아래 형식으로만 출력:
  <A{{i}}>답변</A{{i}}>
- 답이 없으면:
  <A{{i}}>NOT_FOUND</A{{i}}>

[질문 목록]
{questions}

[컨텍스트]
{context}

지금부터 질문 순서대로 답변만 출력하라.
반드시 아래 예시와 동일한 “태그만 있는 형태”로 출력하라.

[예시 — 형식만 참고]
<A1>NOT_FOUND</A1>
<A2>2024년</A2>
<A3>352,000,000</A3>
""".strip()

# -------------------------
# Baseline-compatible utils
# -------------------------
def load_questions_df() -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / "questions.csv")

def get_queries_for_doc(doc_name: str, questions_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """returns [(field, question), ...] ; field == type column in questions.csv (baseline)"""
    common = questions_df[questions_df["doc_id"] == "*"][["type", "question"]]
    per_doc = questions_df[questions_df["doc_id"] == doc_name][["type", "question"]]
    merged = pd.concat([common, per_doc], ignore_index=True)
    return list(zip(merged["type"], merged["question"]))

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
    def generate(self, questions: List[str], context: str) -> List[str]:
        ...


# -------------------------
# Chunkers
# -------------------------
class C1FixedChunker(BaseChunker):
    """Baseline chunk: 800 chars, no overlap"""
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
        # If ALL_DATA doesn't have this doc, fallback to baseline C1 to avoid None issues
        chunks = chunk_from_alldata(doc_path.name, ALL_DATA, size=CONFIG["chunk_length"])
        if chunks is None:
            text = clean_text(extract_text(doc_path))
            s = CONFIG["chunk_length"]
            return [text[i:i+s] for i in range(0, len(text), s)]
        return chunks


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
    """Hybrid: BM25 + Vector, combine scores only for candidate subset to avoid huge RAM/time"""
    def __init__(self, embed_model: SentenceTransformer, bm25_candidates: int = 200):
        self.embed_model = embed_model
        self.bm25_candidates = bm25_candidates

    def build_index(self, chunks: List[str]) -> Any:
        bm25 = rank_bm25.BM25Okapi([c.split() for c in chunks])
        embs = self.embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss_index = faiss.IndexFlatL2(embs.shape[1])
        faiss_index.add(embs.astype("float32"))
        return {"bm25": bm25, "faiss": faiss_index, "chunks": chunks}

    def retrieve(self, index: Any, query_texts: List[str], top_k: int) -> List[int]:
        bm25 = index["bm25"]
        faiss_index = index["faiss"]
        chunks = index["chunks"]

        q_text = " ".join(query_texts)
        # 1) BM25 candidates (cheap)
        bm25_scores = bm25.get_scores(q_text.split())
        cand_n = min(self.bm25_candidates, len(chunks))
        cand_idxs = np.argsort(bm25_scores)[::-1][:cand_n].astype(int)

        # 2) Vector distances only for those candidates
        q_embs = self.embed_model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False)
        q_mean = q_embs.mean(axis=0, keepdims=True).astype("float32")

        # Search top_k over full index is fine too, but we want hybrid rerank:
        # We'll compute vector score for candidate idxs by reconstructing vectors from FAISS is hard for IndexFlatL2.
        # Instead: do FAISS top_k first, then union with BM25 candidates.
        _, vec_I = faiss_index.search(q_mean, min(max(top_k, cand_n), len(chunks)))
        vec_idxs = vec_I[0].astype(int)

        union = np.unique(np.concatenate([cand_idxs, vec_idxs]))
        # Build hybrid score on this union only
        # BM25: higher better
        bm = bm25_scores[union]

        # Vector: we need distances for union -> easiest: do faiss search for len(union)?? not possible by ids directly.
        # Practical compromise: use rank-based score from vec result.
        vec_rank_score = np.zeros(len(chunks), dtype=np.float32)
        for rank, idx in enumerate(vec_idxs, start=1):
            vec_rank_score[idx] = 1.0 / rank  # higher better

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
    
    def generate(self, questions: List[str], context: str) -> List[str]:
        # RFP_PROMPT가 이미 (태그 출력 규칙 포함) 형식 강제를 하고 있다고 가정.
        # 여기서는 prompt 조립 + 모델별 파라미터 호환 + 태그 파싱만 담당.

        q_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        prompt = RFP_PROMPT.format(questions=q_block, context=context)

        is_gpt5 = str(self.model).startswith("gpt-5")

        create_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "RFP 전문 분석가. 형식 엄수."},
                {"role": "user", "content": prompt},
            ],
        }

        # gpt-5 계열: max_completion_tokens 사용, temperature는 기본값(1)만 허용되는 케이스가 있어 미전달 [web:85][web:101]
        if is_gpt5:
            create_kwargs["max_completion_tokens"] = CONFIG["max_completion_tokens"]
        else:
            create_kwargs["max_tokens"] = CONFIG["max_tokens"]
            create_kwargs["temperature"] = CONFIG["temperature"]

        resp = self.client.chat.completions.create(**create_kwargs)
        text = (resp.choices[0].message.content or "")

        # (권장) RFP_PROMPT가 <A1>...</A1> 태그를 강제하는 경우: 태그 파싱
        answers: List[str] = []
        n = len(questions)
        for i in range(1, n + 1):
            start_tag = f"<A{i}>"
            end_tag = f"</A{i}>"
            s = text.find(start_tag)
            e = text.find(end_tag)
            if s != -1 and e != -1 and e > s:
                ans = text[s + len(start_tag): e].strip()
                answers.append(ans if ans else "NOT_FOUND")
            else:
                answers.append("NOT_FOUND")

        return answers

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
    # chunker
    if spec.chunker == "C1":
        chunker = C1FixedChunker(size=CONFIG["chunk_length"])
    elif spec.chunker == "C2":
        chunker = C2PageChunker()
    elif spec.chunker == "C3":
        chunker = C3SectionChunker()
    else:
        raise ValueError(spec.chunker)

    # retriever
    if spec.retriever == "R1":
        retriever = R1BM25Retriever()
    elif spec.retriever == "R2":
        retriever = R2VectorRetriever(embed_model)
    elif spec.retriever == "R3":
        retriever = R3HybridRetriever(embed_model, bm25_candidates=200)
    else:
        raise ValueError(spec.retriever)

    # generator
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

    def run_single_doc_metrics(
        self,
        doc_path: Path,
        gold_fields_df: pd.DataFrame,
        gold_evidence_df: pd.DataFrame,
        top_k: int = 15,
        sim_threshold: int = 80,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFC", doc_path.name)
        queries = get_queries_for_doc(doc_name, self.questions_df)  # [(field, question)]
        q_texts = [q for _, q in queries]

        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)
        idxs = self.retriever.retrieve(index, q_texts, top_k=top_k)

        context = "".join(chunks[i] for i in idxs if 0 <= i < len(chunks))
        answers = self.generator.generate(q_texts, context)

        # --- evaluation (baseline style) ---
        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()

        GOLD_ANCHOR = build_gold_anchor_map(gold_evidence_df)

        # generation score per gold field row
        g_list = []
        for i, (field, _q) in enumerate(queries):
            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None
            pred = answers[i] if i < len(answers) else ""
            g_list.append(eval_gen(pred, gold, threshold=sim_threshold))

        # retrieval score per instance_id (same as baseline notebook you had)
        r_list = []
        for _, row in qdf.iterrows():
            iid = str(row["instance_id"])
            anchors = GOLD_ANCHOR.get(iid, [])
            if anchors:
                r_list.append(eval_retrieval_by_anchor(chunks, idxs, anchors))
            else:
                r_list.append({"recall": np.nan, "mrr": np.nan})

        metrics = {
            "doc_id": doc_name,
            "n_questions": len(qdf),
            "chunk_count": len(chunks),
            "context_length": len(context),

            "ret_recall": float(np.nanmean([x["recall"] for x in r_list])),
            "ret_mrr": float(np.nanmean([x["mrr"] for x in r_list])),

            "gen_fill": float(np.nanmean([x["fill"] for x in g_list])),
            "gen_match": float(np.nanmean([x["match"] for x in g_list])),
            "gen_sim": float(np.nanmean([x["sim"] for x in g_list])),
        }

        # free big objects ASAP (important in notebooks)
        del chunks, index, context, answers
        gc.collect()

        return metrics