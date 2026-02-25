from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocess import rag_experiment as rexp
from preprocess.pp_basic import EVAL_DIR, docs


SPECS: Dict[int, Tuple[str, str, str]] = {
    1: ("C1", "R1", "G1"),
    2: ("C1", "R1", "G2"),
    3: ("C1", "R2", "G1"),
    4: ("C1", "R2", "G2"),
    5: ("C1", "R3", "G1"),
    6: ("C1", "R3", "G2"),
    7: ("C2", "R1", "G1"),
    8: ("C2", "R1", "G2"),
    9: ("C2", "R2", "G1"),
    10: ("C2", "R2", "G2"),
    11: ("C2", "R3", "G1"),
    12: ("C2", "R3", "G2"),
    13: ("C3", "R1", "G1"),
    14: ("C3", "R1", "G2"),
    15: ("C3", "R2", "G1"),
    16: ("C3", "R2", "G2"),
    17: ("C3", "R3", "G1"),
    18: ("C3", "R3", "G2"),
    19: ("C4", "R1", "G1"),
    20: ("C4", "R1", "G2"),
    21: ("C4", "R2", "G1"),
    22: ("C4", "R2", "G2"),
    23: ("C4", "R3", "G1"),
    24: ("C4", "R3", "G2"),
}

NOT_FOUND_TOKENS = {"", "없음", "notfound", "not_found", "gen_fail"}
TARGET_FIELDS = {"project_name", "agency", "duration", "budget"}


def _name_key(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).strip().replace("\ufeff", "")


def _is_not_found(s: str) -> bool:
    return str(s or "").strip().lower() in NOT_FOUND_TOKENS


def _norm_loose(s: str) -> str:
    x = str(s or "").lower()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^\w가-힣]", "", x)
    return x


def _looks_format_mismatch(pred: str, gold: str, sim: float) -> bool:
    if _is_not_found(pred) or _is_not_found(gold):
        return False
    pn = _norm_loose(pred)
    gn = _norm_loose(gold)
    if not pn or not gn:
        return False
    if pn == gn or pn in gn or gn in pn:
        return True
    return float(sim) >= 60.0


def _find_hit_rank(chunks: List[str], idxs: List[int], anchors: List[str]) -> Optional[int]:
    if not anchors:
        return None
    for rank, ci in enumerate(idxs, start=1):
        if 0 <= int(ci) < len(chunks):
            c = chunks[int(ci)]
            if any(a in c for a in anchors):
                return rank
    return None


def _anchor_in_doc_joined(joined: str, anchors: List[str], cache: Dict[str, bool]) -> bool:
    if not anchors:
        return False
    for a in anchors:
        if a not in cache:
            cache[a] = a in joined
        if cache[a]:
            return True
    return False


def _load_gold_fields_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = str(rec["instance_id"])
            doc_id = _name_key(rec["doc_id"])
            fields = rec.get("fields", {}) or {}
            for k, v in fields.items():
                if isinstance(v, list):
                    v = " ".join(str(x) for x in v)
                rows.append(
                    {
                        "instance_id": iid,
                        "doc_id": doc_id,
                        "field": str(k),
                        "gold": str(v),
                    }
                )
    return pd.DataFrame(rows)


def _load_gold_evidence_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "instance_id": str(rec["instance_id"]),
                    "doc_id": _name_key(rec["doc_id"]),
                    "anchor_text": str(rec.get("anchor_text", "") or "").strip(),
                }
            )
    return pd.DataFrame(rows)


def _load_pred_maps(exp_id: int, base_dir: Path) -> Dict[str, Dict[str, str]]:
    pred_dir = base_dir / "outputs" / f"exp{exp_id:02d}_pred_maps"
    out: Dict[str, Dict[str, str]] = {}
    for fp in sorted(pred_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            out[_name_key(fp.stem)] = json.load(f)
    return out


def _make_chunker(code: str):
    if code == "C1":
        return rexp.C1FixedChunker(size=rexp.CONFIG["chunk_length"])
    if code == "C2":
        return rexp.C2PageChunker()
    if code == "C3":
        return rexp.C3SectionChunker()
    if code == "C4":
        return rexp.C4DoclingChunker()
    raise ValueError(code)


def _make_retriever(code: str, embed_model: SentenceTransformer):
    if code == "R1":
        return rexp.R1BM25Retriever()
    if code == "R2":
        return rexp.R2VectorRetriever(embed_model)
    if code == "R3":
        return rexp.R3HybridRetriever(embed_model, bm25_candidates=200)
    if code == "R4":
        return rexp.R4RerankerRetriever(embed_model)
    raise ValueError(code)


def _nanmean(s: pd.Series) -> float:
    arr = s.astype(float).to_numpy()
    if arr.size == 0 or np.isnan(arr).all():
        return float(np.nan)
    return float(np.nanmean(arr))


def _bin_hit_rank(r: Optional[float]) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "miss"
    rr = int(r)
    if rr == 1:
        return "r1"
    if rr == 2:
        return "r2"
    if rr <= 5:
        return "r3_5"
    if rr <= 10:
        return "r6_10"
    return "r11_plus"


def _to_md_table(headers: List[str], rows: List[List[Any]]) -> str:
    def _s(x: Any) -> str:
        if isinstance(x, float):
            if np.isnan(x):
                return ""
            return f"{x:.4f}"
        return str(x)

    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_s(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def _tag_failure(
    match: float,
    pred: str,
    gold: str,
    sim: float,
    strict_applied: float,
    strict_match: float,
    has_anchor: bool,
    anchor_in_doc: bool,
    hit_rank: Optional[int],
) -> Tuple[str, str]:
    if not np.isfinite(match) or float(match) != 0.0:
        return "", ""

    if not has_anchor:
        return "gold노이즈", "anchor_text 없음"
    if not anchor_in_doc:
        return "gold노이즈", "anchor_text가 전체 청크에서 발견되지 않음"
    if hit_rank is None:
        return "검색실패", "top-k 내 anchor 미검출"
    if _is_not_found(pred):
        return "추출실패", "검색 성공 후 NOT_FOUND/GEN_FAIL 반환"
    if float(strict_applied) == 1.0 and np.isfinite(strict_match) and float(strict_match) == 0.0:
        return "추출실패", "strict 필드 값 불일치"
    if _is_not_found(gold) and not _is_not_found(pred):
        return "gold노이즈", "gold=NOT_FOUND인데 모델이 내용 생성"
    if _looks_format_mismatch(pred, gold, sim):
        return "포맷불일치", "정규화/유사도 기준 근접"
    return "추출실패", "검색 성공했지만 값 추출/매핑 실패"


def analyze_exp(
    exp_id: int,
    top_k: int,
    sim_threshold: int,
    base_dir: Path,
    out_dir: Path,
    embed_model: SentenceTransformer,
    questions_df: pd.DataFrame,
    gold_fields_df: pd.DataFrame,
    gold_evidence_df: pd.DataFrame,
    limit_docs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exp_id not in SPECS:
        raise ValueError(f"Unknown exp_id: {exp_id}")

    chunk_code, retr_code, _gen_code = SPECS[exp_id]
    chunker = _make_chunker(chunk_code)
    retriever = _make_retriever(retr_code, embed_model)
    pred_maps = _load_pred_maps(exp_id, base_dir)

    gold_doc_keys = set(_name_key(x) for x in gold_fields_df["doc_id"].astype(str).unique())
    eval_docs = [p for p in docs if _name_key(p.name) in gold_doc_keys]
    if limit_docs is not None and limit_docs > 0:
        eval_docs = eval_docs[:limit_docs]
    anchor_map = rexp.build_gold_anchor_map(gold_evidence_df)

    rows: List[Dict[str, Any]] = []

    for di, doc_path in enumerate(eval_docs, start=1):
        doc_name = _name_key(doc_path.name)
        queries = rexp.get_queries_for_doc(doc_name, questions_df)
        chunks = chunker.chunk(doc_path)
        index = retriever.build_index(chunks)
        joined_doc = "\n".join(chunks)
        anchor_presence_cache: Dict[str, bool] = {}
        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()
        pmap = pred_maps.get(doc_name, {})

        # R2는 질문 임베딩을 문서 단위로 배치 처리해서 속도 개선
        idx_cache: Dict[Tuple[str, str], List[int]] = {}
        if retr_code == "R2" and hasattr(retriever, "embed_model"):
            q_texts = [q for _, q in queries]
            if q_texts:
                q_embs = retriever.embed_model.encode(q_texts, convert_to_numpy=True, show_progress_bar=False)
                for (field, question), q_emb in zip(queries, q_embs):
                    _, I = index.search(q_emb.reshape(1, -1).astype("float32"), top_k)
                    idx_cache[(field, question)] = [int(i) for i in I[0]]

        for field, question in queries:
            if (field, question) in idx_cache:
                idxs = idx_cache[(field, question)]
            else:
                idxs = retriever.retrieve(index, [question], top_k=top_k)
            pred = str(pmap.get(field, "") or "").strip()
            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None
            g = rexp.eval_gen(pred, gold, threshold=sim_threshold, field=field)

            for _, r in gold_row.iterrows():
                iid = str(r["instance_id"])
                anchors = [a for a in anchor_map.get(iid, []) if str(a).strip()]
                has_anchor = bool(anchors)
                in_doc = _anchor_in_doc_joined(joined_doc, anchors, anchor_presence_cache) if has_anchor else False
                hit_rank = _find_hit_rank(chunks, idxs, anchors) if has_anchor else None

                tag, reason = _tag_failure(
                    match=float(g["match"]) if np.isfinite(g["match"]) else np.nan,
                    pred=pred,
                    gold=str(gold or ""),
                    sim=float(g["sim"]) if np.isfinite(g["sim"]) else np.nan,
                    strict_applied=float(g["strict_applied"]),
                    strict_match=float(g["strict_match"]) if np.isfinite(g["strict_match"]) else np.nan,
                    has_anchor=has_anchor,
                    anchor_in_doc=in_doc,
                    hit_rank=hit_rank,
                )

                rows.append(
                    {
                        "exp_id": exp_id,
                        "doc_id": doc_name,
                        "instance_id": iid,
                        "field": str(field),
                        "question": str(question),
                        "gold": str(gold or ""),
                        "pred": pred,
                        "fill": float(g["fill"]),
                        "match": float(g["match"]) if np.isfinite(g["match"]) else np.nan,
                        "sim": float(g["sim"]) if np.isfinite(g["sim"]) else np.nan,
                        "strict_match": float(g["strict_match"]) if np.isfinite(g["strict_match"]) else np.nan,
                        "strict_applied": float(g["strict_applied"]),
                        "has_anchor": has_anchor,
                        "anchor_in_doc": in_doc,
                        "hit_rank": hit_rank,
                        "tag": tag,
                        "tag_reason": reason,
                    }
                )

        print(f"[exp {exp_id}] {di}/{len(eval_docs)} docs done: {doc_name}", flush=True)

    detail_df = pd.DataFrame(rows)

    fail_df = detail_df[(detail_df["match"] == 0.0)].copy()
    fail_df = fail_df.sort_values(["tag", "field", "doc_id", "instance_id"]).reset_index(drop=True)

    fmask = detail_df["field"].astype(str).isin(TARGET_FIELDS)
    field_df = (
        detail_df[fmask]
        .groupby("field", dropna=False)
        .apply(
            lambda d: pd.Series(
                {
                    "n": int(len(d)),
                    "match": _nanmean(d["match"]),
                    "sim": _nanmean(d["sim"]),
                    "strict_match": _nanmean(d["strict_match"]),
                    "strict_coverage": _nanmean(d["strict_applied"]),
                }
            )
        )
        .reset_index()
        .sort_values("field")
    )
    field_df.insert(0, "exp_id", exp_id)

    hmask = detail_df["has_anchor"] & detail_df["anchor_in_doc"]
    hit_df = detail_df[hmask].copy()
    hit_df["rank_bin"] = hit_df["hit_rank"].map(_bin_hit_rank)
    hit_dist = (
        hit_df.groupby("rank_bin")
        .size()
        .reindex(["r1", "r2", "r3_5", "r6_10", "r11_plus", "miss"], fill_value=0)
        .reset_index(name="count")
    )
    hit_dist["ratio"] = hit_dist["count"] / max(int(hit_dist["count"].sum()), 1)
    hit_dist.insert(0, "exp_id", exp_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(out_dir / f"exp{exp_id:02d}_detail_eval.csv", index=False, encoding="utf-8-sig")
    fail_df.to_csv(out_dir / f"exp{exp_id:02d}_failure_tags.csv", index=False, encoding="utf-8-sig")
    field_df.to_csv(out_dir / f"exp{exp_id:02d}_field_breakdown.csv", index=False, encoding="utf-8-sig")
    hit_dist.to_csv(out_dir / f"exp{exp_id:02d}_hit_rank_dist.csv", index=False, encoding="utf-8-sig")

    return detail_df, field_df, hit_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-ids", nargs="+", type=int, default=[3, 21])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sim-threshold", type=int, default=80)
    parser.add_argument("--embed-model", type=str, default="nlpai-lab/KoE5")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--md-name", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / args.out_dir

    questions_df = rexp.load_questions_df()
    gold_fields_df = _load_gold_fields_jsonl(EVAL_DIR / "gold_fields.jsonl")
    gold_evidence_df = _load_gold_evidence_jsonl(EVAL_DIR / "gold_evidence.jsonl")
    embed_model = SentenceTransformer(args.embed_model)

    all_fields: List[pd.DataFrame] = []
    all_hits: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, Any]] = []

    for exp_id in args.exp_ids:
        detail_df, field_df, hit_dist = analyze_exp(
            exp_id=exp_id,
            top_k=args.top_k,
            sim_threshold=args.sim_threshold,
            base_dir=base_dir,
            out_dir=out_dir,
            embed_model=embed_model,
            questions_df=questions_df,
            gold_fields_df=gold_fields_df,
            gold_evidence_df=gold_evidence_df,
            limit_docs=args.limit_docs,
        )
        all_fields.append(field_df)
        all_hits.append(hit_dist)

        fail_df = detail_df[detail_df["match"] == 0.0]
        tag_counts = fail_df["tag"].value_counts(dropna=False).to_dict()
        print(
            f"[exp {exp_id}] n={len(detail_df)} fail={len(fail_df)} "
            f"tags={tag_counts}"
        )
        fail_by_field = fail_df.groupby("field").size().sort_values(ascending=False).head(10)
        all_summaries.append(
            {
                "exp_id": exp_id,
                "n": int(len(detail_df)),
                "fail": int(len(fail_df)),
                "tag_counts": tag_counts,
                "fail_by_field": fail_by_field,
            }
        )

    if len(all_fields) >= 2:
        cmp_fields = pd.concat(all_fields, ignore_index=True)
        cmp_fields.to_csv(out_dir / "field_breakdown_compare.csv", index=False, encoding="utf-8-sig")
    if len(all_hits) >= 2:
        cmp_hits = pd.concat(all_hits, ignore_index=True)
        cmp_hits.to_csv(out_dir / "hit_rank_dist_compare.csv", index=False, encoding="utf-8-sig")

    md_name = args.md_name.strip() or f"analysis_summary_exp{'_'.join(f'{x:02d}' for x in args.exp_ids)}.md"
    md_path = out_dir / md_name
    lines: List[str] = []
    lines.append("# Failure Analysis Summary")
    lines.append("")
    lines.append(f"- exp_ids: {args.exp_ids}")
    lines.append(f"- top_k: {args.top_k}")
    lines.append(f"- sim_threshold: {args.sim_threshold}")
    if args.limit_docs:
        lines.append(f"- limit_docs: {args.limit_docs}")
    lines.append("")

    for s in all_summaries:
        lines.append(f"## exp {s['exp_id']}")
        lines.append("")
        lines.append(f"- total rows: {s['n']}")
        lines.append(f"- fail rows (match=0): {s['fail']}")
        tag_rows = [[k, v] for k, v in s["tag_counts"].items()]
        if tag_rows:
            lines.append("")
            lines.append("### tag counts")
            lines.append("")
            lines.append(_to_md_table(["tag", "count"], tag_rows))
        fbf = s["fail_by_field"]
        if len(fbf) > 0:
            lines.append("")
            lines.append("### fail by field (top10)")
            lines.append("")
            lines.append(_to_md_table(["field", "count"], [[idx, int(val)] for idx, val in fbf.items()]))
        lines.append("")

    if len(all_fields) >= 2:
        cmp_fields = pd.concat(all_fields, ignore_index=True)
        lines.append("## field breakdown compare")
        lines.append("")
        rows = [
            [
                int(r["exp_id"]),
                r["field"],
                int(r["n"]),
                float(r["match"]),
                float(r["sim"]),
                float(r["strict_match"]) if pd.notna(r["strict_match"]) else np.nan,
                float(r["strict_coverage"]),
            ]
            for _, r in cmp_fields.iterrows()
        ]
        lines.append(
            _to_md_table(
                ["exp_id", "field", "n", "match", "sim", "strict_match", "strict_coverage"],
                rows,
            )
        )
        lines.append("")

    if len(all_hits) >= 2:
        cmp_hits = pd.concat(all_hits, ignore_index=True)
        lines.append("## hit rank distribution compare")
        lines.append("")
        rows = [
            [int(r["exp_id"]), r["rank_bin"], int(r["count"]), float(r["ratio"])]
            for _, r in cmp_hits.iterrows()
        ]
        lines.append(_to_md_table(["exp_id", "rank_bin", "count", "ratio"], rows))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown summary: {md_path}")
    print(f"Saved analysis files to: {out_dir}")


if __name__ == "__main__":
    main()
