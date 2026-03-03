from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import hashlib
import pickle
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .loaders import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _artifact_signature(artifact_path: Path) -> str:
    st = artifact_path.stat()
    raw = f"{artifact_path.resolve()}||{st.st_mtime_ns}||{st.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _index_path(index_dir: Path, doc_id: str, source: str, artifact_sig: str) -> Path:
    safe = doc_id.replace("/", "_")
    return index_dir / f"{safe}__src={source}__sig={artifact_sig}.pkl"


# --- BM25 (간단 구현: 한국어도 공백 토큰으로 우선) ---
_TOKEN_RE = re.compile(r"\S+")

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

class BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        # df
        df = {}
        for toks in corpus_tokens:
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.df = df

        # idf
        self.idf = {}
        for t, n in df.items():
            # okapi idf
            self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

        # tf per doc as dict for speed
        self.tfs = []
        for toks in corpus_tokens:
            d = {}
            for t in toks:
                d[t] = d.get(t, 0) + 1
            self.tfs.append(d)

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        if self.N == 0:
            return np.zeros(0, dtype=np.float32)
        scores = np.zeros(self.N, dtype=np.float32)
        q = query_tokens
        for i in range(self.N):
            dl = self.doc_lens[i] or 1
            denom_base = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            tf = self.tfs[i]
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + denom_base)
            scores[i] = s
        return scores


class CachedHybridRetriever:
    """
    TF-IDF cosine + BM25 하이브리드
    score = alpha * tfidf_norm + (1-alpha) * bm25_norm
    """
    def __init__(self, chunks: List[Chunk], vec: TfidfVectorizer, mat, bm25: BM25):
        self.chunks = chunks
        self.vec = vec
        self.mat = mat
        self.bm25 = bm25

    def search(self, query: str, k: int = 6, alpha: float = 0.7) -> List[ScoredChunk]:
        q = (query or "").strip()
        if not q:
            return []

        # TF-IDF
        qv = self.vec.transform([q])
        tfidf = cosine_similarity(qv, self.mat).ravel().astype(np.float32)

        # BM25
        bm25 = self.bm25.scores(_tokenize(q))

        # normalize to 0..1
        def norm(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            mn, mx = float(x.min()), float(x.max())
            if mx - mn < 1e-8:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        tfidf_n = norm(tfidf)
        bm25_n = norm(bm25)

        score = alpha * tfidf_n + (1 - alpha) * bm25_n

        topk = np.argsort(-score)[:k]
        out: List[ScoredChunk] = []
        for idx in topk:
            out.append(ScoredChunk(self.chunks[int(idx)], float(score[int(idx)])))
        return out


def build_or_load_hybrid(
    chunks: List[Chunk],
    index_dir: Path,
    doc_id: str,
    source: str,
    artifact_path: Path,
) -> CachedHybridRetriever:
    index_dir.mkdir(parents=True, exist_ok=True)
    sig = _artifact_signature(artifact_path)
    ipath = _index_path(index_dir, doc_id, source + "__hybrid", sig)

    if ipath.exists():
        with open(ipath, "rb") as f:
            return pickle.load(f)

    texts = [c.text for c in chunks]
    vec = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=200_000)
    mat = vec.fit_transform(texts)

    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25(corpus_tokens)

    retriever = CachedHybridRetriever(chunks=chunks, vec=vec, mat=mat, bm25=bm25)
    with open(ipath, "wb") as f:
        pickle.dump(retriever, f)
    return retriever


def pages_from_results(results: List[ScoredChunk]) -> List[int]:
    return sorted(set([r.chunk.page for r in results]))


def evidence_text(results: List[ScoredChunk], max_chars: int = 2000) -> str:
    lines = []
    for rank, r in enumerate(results, start=1):
        t = r.chunk.text
        if len(t) > max_chars:
            t = t[:max_chars] + "…"
        sec = f" | section={r.chunk.section_path}" if r.chunk.section_path else ""
        lines.append(
            f"[{rank}] (page={r.chunk.page}, chunk_id={r.chunk.chunk_id}, score={r.score:.4f}{sec})\n{t}\n"
        )
    return "\n".join(lines)