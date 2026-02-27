from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from preprocess import rag_experiment as rexp
    from preprocess.pp_basic import EVAL_DIR, docs
except ModuleNotFoundError:
    # allow direct execution: `uv run python notebooks/preprocess/run_experiments.py ...`
    NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]
    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))
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

TARGET_FIELDS = ["project_name", "agency", "duration", "budget", "requirements_must", "eligibility", "eval_items", "purpose", "contract_type"]


def _name_key(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).strip().replace("\ufeff", "")


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


def _nanmean_safe(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float(np.nan)
    return float(np.nanmean(arr))


def _build_eval_docs(gold_fields_df: pd.DataFrame, n_docs: Optional[int]) -> List[Path]:
    gold_doc_keys = set(_name_key(x) for x in gold_fields_df["doc_id"].astype(str).unique())
    eval_docs = [p for p in docs if _name_key(p.name) in gold_doc_keys]
    if n_docs is not None and n_docs > 0:
        eval_docs = eval_docs[:n_docs]
    return eval_docs


def _apply_config(
    args: argparse.Namespace,
    retrieve_k: int,
    context_k: int,
    recall_k: int,
) -> Tuple[int, int, int]:
    rk = int(max(1, retrieve_k))
    ck = int(max(1, min(context_k, rk)))
    rck = int(max(1, min(recall_k, rk)))
    rexp.CONFIG["retrieve_k"] = rk
    rexp.CONFIG["context_k"] = ck
    rexp.CONFIG["recall_k"] = rck
    rexp.CONFIG["top_k"] = rk  # backward compatibility
    rexp.CONFIG["max_context_chars"] = int(args.max_context_chars)
    rexp.CONFIG["max_context_chars_per_question"] = int(args.max_context_chars_per_question)
    rexp.CONFIG["target_field_max_context_chars"] = int(args.target_field_max_context_chars)
    return rk, ck, rck


def _to_md_table(headers: List[str], rows: List[List[Any]]) -> str:
    def _fmt(x: Any) -> str:
        if isinstance(x, float):
            if np.isnan(x):
                return ""
            return f"{x:.4f}"
        return str(x)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows)
    return "\n".join(lines)


def _write_exp_insight_md(
    exp_id: int,
    exp_df: pd.DataFrame,
    doc_df: pd.DataFrame,
    field_df: pd.DataFrame,
    insight_path: Path,
    retrieve_k: int,
    context_k: int,
    recall_k: int,
    analysis_extra: Optional[Dict[str, Any]] = None,
) -> None:
    insight_path.parent.mkdir(parents=True, exist_ok=True)
    row = exp_df.iloc[0].to_dict() if not exp_df.empty else {}

    metrics_cols = [
        "ret_recall",
        "ret_mrr",
        "hit_rank_mean",
        "hit_r1_ratio",
        "hit_r2_ratio",
        "hit_r1_r2_ratio",
        "gen_fill",
        "gen_match",
        "gen_sim",
        "gen_match_strict",
        "gen_strict_coverage",
    ]

    lines: List[str] = []
    lines.append(f"# Exp {exp_id} Insight")
    lines.append("")
    lines.append(f"- retrieve_k: {retrieve_k}")
    lines.append(f"- context_k: {context_k}")
    lines.append(f"- recall_k: {recall_k}")
    lines.append(f"- n_docs: {int(row.get('n_docs', len(doc_df)) or len(doc_df))}")
    lines.append("")

    metric_rows: List[List[Any]] = []
    for c in metrics_cols:
        if c in row:
            metric_rows.append([c, float(row[c]) if pd.notna(row[c]) else np.nan])
    if metric_rows:
        lines.append("## Summary Metrics")
        lines.append("")
        lines.append(_to_md_table(["metric", "value"], metric_rows))
        lines.append("")

    if not field_df.empty and set(["field", "match", "sim", "strict_match", "strict_applied"]).issubset(field_df.columns):
        f = field_df.copy()
        f = f[f["field"].astype(str).isin(TARGET_FIELDS)]
        if not f.empty:
            grp = (
                f.groupby("field", dropna=False)
                .agg(
                    n=("field", "size"),
                    match=("match", "mean"),
                    sim=("sim", "mean"),
                    strict_match=("strict_match", "mean"),
                    strict_coverage=("strict_applied", "mean"),
                )
                .reset_index()
                .sort_values("field")
            )
            lines.append("## Field Breakdown")
            lines.append("")
            rows = [
                [
                    str(r["field"]),
                    int(r["n"]),
                    float(r["match"]) if pd.notna(r["match"]) else np.nan,
                    float(r["sim"]) if pd.notna(r["sim"]) else np.nan,
                    float(r["strict_match"]) if pd.notna(r["strict_match"]) else np.nan,
                    float(r["strict_coverage"]) if pd.notna(r["strict_coverage"]) else np.nan,
                ]
                for _, r in grp.iterrows()
            ]
            lines.append(_to_md_table(["field", "n", "match", "sim", "strict_match", "strict_coverage"], rows))
            lines.append("")

    if not doc_df.empty and "gen_match" in doc_df.columns:
        bad = doc_df.sort_values(["gen_match", "gen_sim"], ascending=[True, True]).head(5)
        if not bad.empty:
            lines.append("## Worst Docs (Top 5)")
            lines.append("")
            rows = []
            for _, r in bad.iterrows():
                rows.append(
                    [
                        str(r.get("doc_id", "")),
                        float(r.get("gen_match", np.nan)),
                        float(r.get("gen_sim", np.nan)),
                        float(r.get("ret_recall", np.nan)),
                        float(r.get("ret_mrr", np.nan)),
                    ]
                )
            lines.append(_to_md_table(["doc_id", "gen_match", "gen_sim", "ret_recall", "ret_mrr"], rows))
            lines.append("")

    if analysis_extra:
        tag_counts = analysis_extra.get("tag_counts", {}) or {}
        if tag_counts:
            lines.append("## Failure Tags")
            lines.append("")
            rows = [[k, int(v)] for k, v in tag_counts.items()]
            lines.append(_to_md_table(["tag", "count"], rows))
            lines.append("")
        fail_by_field = analysis_extra.get("fail_by_field")
        if isinstance(fail_by_field, pd.Series) and len(fail_by_field) > 0:
            lines.append("## Fail By Field (Top10)")
            lines.append("")
            rows = [[str(k), int(v)] for k, v in fail_by_field.items()]
            lines.append(_to_md_table(["field", "count"], rows))
            lines.append("")
        hit_df = analysis_extra.get("hit_dist")
        if isinstance(hit_df, pd.DataFrame) and not hit_df.empty:
            lines.append("## Hit Rank Dist")
            lines.append("")
            rows = [[str(r["rank_bin"]), int(r["count"]), float(r["ratio"])] for _, r in hit_df.iterrows()]
            lines.append(_to_md_table(["rank_bin", "count", "ratio"], rows))
            lines.append("")

    insight_path.write_text("\n".join(lines), encoding="utf-8")


def run_one_exp(
    exp_id: int,
    eval_docs: List[Path],
    gold_fields_df: pd.DataFrame,
    gold_evidence_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    embed_model: SentenceTransformer,
    client: OpenAI,
    out_dir: Path,
    sim_threshold: int,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exp_id not in SPECS:
        raise ValueError(f"Unknown exp_id={exp_id}")

    c, r, g = SPECS[exp_id]
    spec = rexp.ExperimentSpec(exp_id=exp_id, chunker=c, retriever=r, generator=g)
    chunker, retriever, generator = rexp.make_components(spec, embed_model=embed_model, client=client)
    rag = rexp.RAGExperiment(chunker=chunker, retriever=retriever, generator=generator, questions_df=questions_df)

    if save_artifacts:
        pred_map_dir = out_dir / f"exp{exp_id:02d}_pred_maps"
        pred_map_dir.mkdir(parents=True, exist_ok=True)
    else:
        pred_map_dir = None

    rows: List[Dict[str, Any]] = []
    field_rows: List[Dict[str, Any]] = []

    for doc_path in tqdm(eval_docs, desc=f"Exp {exp_id} docs"):
        m = rag.run_single_doc_metrics_singleq(
            doc_path,
            gold_fields_df=gold_fields_df,
            gold_evidence_df=gold_evidence_df,
            top_k=int(rexp.CONFIG.get("top_k", 20)),
            sim_threshold=sim_threshold,
        )
        m["exp_id"] = spec.exp_id
        m["chunker"] = spec.chunker
        m["retriever"] = spec.retriever
        m["generator"] = spec.generator
        rows.append(m)

        doc_id = _name_key(doc_path.name)
        if pred_map_dir is not None:
            pmap = m.get("pred_map", {}) or {}
            with open(pred_map_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
                json.dump(pmap, f, ensure_ascii=False, indent=2)

        for fr in (m.get("field_rows") or []):
            one = dict(fr)
            one["doc_id"] = doc_id
            one["exp_id"] = exp_id
            one["chunker"] = spec.chunker
            one["retriever"] = spec.retriever
            one["generator"] = spec.generator
            field_rows.append(one)

    doc_df = pd.DataFrame(rows)
    field_df = pd.DataFrame(field_rows)

    metric_cols = [
        "ret_recall",
        "ret_mrr",
        "hit_rank_mean",
        "hit_r1_ratio",
        "hit_r2_ratio",
        "hit_r1_r2_ratio",
        "gen_fill",
        "gen_match",
        "gen_sim",
        "gen_match_strict",
        "gen_strict_coverage",
    ]
    metric_cols = [c for c in metric_cols if c in doc_df.columns]
    avg = {k: _nanmean_safe(doc_df[k].tolist()) for k in metric_cols}
    exp_df = pd.DataFrame(
        [
            {
                "exp_id": spec.exp_id,
                "chunk": spec.chunker,
                "retriever": spec.retriever,
                "model": spec.generator,
                "n_docs": len(doc_df),
                **avg,
            }
        ]
    )

    if save_artifacts:
        exp_df.to_csv(out_dir / f"exp{exp_id:02d}_explevel.csv", index=False, encoding="utf-8-sig")
        doc_df.to_csv(out_dir / f"exp{exp_id:02d}_doclevel.csv", index=False, encoding="utf-8-sig")
        if not field_df.empty:
            field_df.to_csv(out_dir / f"exp{exp_id:02d}_fieldlevel.csv", index=False, encoding="utf-8-sig")

    return exp_df, doc_df, field_df


def _run_k_sweep(
    args: argparse.Namespace,
    eval_docs: List[Path],
    gold_fields_df: pd.DataFrame,
    gold_evidence_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    embed_model: SentenceTransformer,
    client: OpenAI,
    out_dir: Path,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    objective = str(args.sweep_objective)
    metric_tiebreak = ["gen_match", "gen_sim", "ret_mrr", "ret_recall", "hit_r1_r2_ratio"]

    print(
        f"SWEEP START | exp_id={args.sweep_exp_id} "
        f"retrieve_grid={args.retrieve_k_grid} context_grid={args.context_k_grid} "
        f"recall_k={args.recall_k}"
    )
    for rk in args.retrieve_k_grid:
        for ck in args.context_k_grid:
            if int(ck) > int(rk):
                continue
            rk_now, ck_now, rck_now = _apply_config(args, int(rk), int(ck), int(args.recall_k))
            exp_df, _, _ = run_one_exp(
                exp_id=int(args.sweep_exp_id),
                eval_docs=eval_docs,
                gold_fields_df=gold_fields_df,
                gold_evidence_df=gold_evidence_df,
                questions_df=questions_df,
                embed_model=embed_model,
                client=client,
                out_dir=out_dir,
                sim_threshold=int(args.sim_threshold),
                save_artifacts=False,
            )
            r = exp_df.iloc[0].to_dict()
            r["retrieve_k"] = rk_now
            r["context_k"] = ck_now
            r["recall_k"] = rck_now
            r["objective_value"] = float(r.get(objective, np.nan)) if objective in r else np.nan
            rows.append(r)
            print(
                f"SWEEP | retrieve_k={rk_now} context_k={ck_now} recall_k={rck_now} "
                f"{objective}={r.get(objective, np.nan)}"
            )

    sweep_df = pd.DataFrame(rows)
    if sweep_df.empty:
        return sweep_df, None

    sort_cols = [objective] + [c for c in metric_tiebreak if c in sweep_df.columns and c != objective]
    ascending = [objective == "hit_rank_mean"] + [False] * (len(sort_cols) - 1)
    sweep_sorted = sweep_df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)

    out_csv = out_dir / f"k_sweep_exp{int(args.sweep_exp_id):02d}.csv"
    sweep_sorted.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved sweep results: {out_csv}")

    best = sweep_sorted.iloc[0].to_dict()
    best_json_path = out_dir / f"k_sweep_best_exp{int(args.sweep_exp_id):02d}.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(
        f"SWEEP BEST | retrieve_k={int(best['retrieve_k'])} context_k={int(best['context_k'])} "
        f"recall_k={int(best.get('recall_k', args.recall_k))} "
        f"{objective}={best.get(objective)}"
    )
    print(f"Saved best config: {best_json_path}")
    return sweep_sorted, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-ids", nargs="+", type=int, default=[3, 21])
    parser.add_argument("--n-docs", type=int, default=None)
    parser.add_argument("--embed-model", type=str, default="nlpai-lab/KoE5")
    parser.add_argument("--sim-threshold", type=int, default=80)

    parser.add_argument("--retrieve-k", type=int, default=9)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-context-chars-per-question", type=int, default=3000)
    parser.add_argument("--target-field-max-context-chars", type=int, default=2200)

    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--analysis-out-dir", type=str, default="outputs/analysis")
    parser.add_argument("--analysis-md-name", type=str, default="analysis_summary.md")
    parser.add_argument("--run-tag", type=str, default="")

    parser.add_argument("--sweep-k", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--sweep-exp-id", type=int, default=3)
    parser.add_argument("--retrieve-k-grid", nargs="+", type=int, default=[20, 24, 28, 30])
    parser.add_argument("--context-k-grid", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument(
        "--sweep-objective",
        type=str,
        default="gen_match",
        choices=[
            "ret_recall",
            "ret_mrr",
            "hit_rank_mean",
            "hit_r1_ratio",
            "hit_r2_ratio",
            "hit_r1_r2_ratio",
            "gen_fill",
            "gen_match",
            "gen_sim",
            "gen_match_strict",
            "gen_strict_coverage",
        ],
    )
    parser.add_argument("--no-use-best-k", action="store_true", help="Do not apply sweep best k to main run")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--insight-every-exp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / args.out_dir
    analysis_dir = base_dir / args.analysis_out_dir
    if args.run_tag.strip():
        out_dir = out_dir / args.run_tag.strip()
        analysis_dir = analysis_dir / args.run_tag.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    gold_fields_df = _load_gold_fields_jsonl(EVAL_DIR / "gold_fields.jsonl")
    gold_evidence_df = _load_gold_evidence_jsonl(EVAL_DIR / "gold_evidence.jsonl")
    questions_df = rexp.load_questions_df()
    eval_docs = _build_eval_docs(gold_fields_df, args.n_docs)

    print(f"DOCS | n_docs={len(eval_docs)}")

    embed_model = SentenceTransformer(args.embed_model)
    client = OpenAI()

    selected_retrieve_k = int(args.retrieve_k)
    selected_context_k = int(args.context_k)
    selected_recall_k = int(args.recall_k)

    if args.sweep_k:
        sweep_df, best = _run_k_sweep(
            args=args,
            eval_docs=eval_docs,
            gold_fields_df=gold_fields_df,
            gold_evidence_df=gold_evidence_df,
            questions_df=questions_df,
            embed_model=embed_model,
            client=client,
            out_dir=out_dir,
        )
        if sweep_df.empty:
            raise RuntimeError("Sweep returned no rows. Check sweep grid values.")

        if best is not None and not args.no_use_best_k:
            selected_retrieve_k = int(best["retrieve_k"])
            selected_context_k = int(best["context_k"])
            print(
                f"APPLY BEST K | retrieve_k={selected_retrieve_k} "
                f"context_k={selected_context_k} recall_k={selected_recall_k}"
            )

        if args.sweep_only:
            print("Sweep-only mode finished.")
            return

    rk, ck, rck = _apply_config(args, selected_retrieve_k, selected_context_k, selected_recall_k)
    print(
        f"RUN CONFIG | exp_ids={args.exp_ids} n_docs={len(eval_docs)} "
        f"retrieve_k={rk} context_k={ck} recall_k={rck}"
    )

    if args.run_analysis:
        from preprocess import analyze_failure_cases as af

    exp_rows: List[pd.DataFrame] = []
    all_fields: List[pd.DataFrame] = []
    all_hits: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, Any]] = []

    for exp_id in args.exp_ids:
        exp_df, doc_df, field_df = run_one_exp(
            exp_id=exp_id,
            eval_docs=eval_docs,
            gold_fields_df=gold_fields_df,
            gold_evidence_df=gold_evidence_df,
            questions_df=questions_df,
            embed_model=embed_model,
            client=client,
            out_dir=out_dir,
            sim_threshold=int(args.sim_threshold),
            save_artifacts=True,
        )
        exp_rows.append(exp_df)
        print(f"Saved exp outputs: exp{int(exp_id):02d}_*.csv, exp{int(exp_id):02d}_pred_maps/")

        analysis_extra: Optional[Dict[str, Any]] = None
        if args.run_analysis:
            detail_df, breakdown_df, hit_df = af.analyze_exp(
                exp_id=exp_id,
                retrieve_k=rk,
                sim_threshold=int(args.sim_threshold),
                base_dir=base_dir,
                out_dir=analysis_dir,
                embed_model=embed_model,
                questions_df=questions_df,
                gold_fields_df=gold_fields_df,
                gold_evidence_df=gold_evidence_df,
                limit_docs=args.n_docs,
                pred_maps_root=out_dir,
            )
            fail_df = detail_df[detail_df["match"] == 0.0]
            fail_by_field = fail_df.groupby("field").size().sort_values(ascending=False).head(10)
            tag_counts = fail_df["tag"].value_counts(dropna=False).to_dict()
            analysis_extra = {
                "tag_counts": tag_counts,
                "fail_by_field": fail_by_field,
                "hit_dist": hit_df,
            }
            all_fields.append(breakdown_df)
            all_hits.append(hit_df)
            all_summaries.append(
                {
                    "exp_id": exp_id,
                    "n": int(len(detail_df)),
                    "fail": int(len(fail_df)),
                    "tag_counts": tag_counts,
                    "fail_by_field": fail_by_field,
                }
            )

        if args.insight_every_exp:
            insight_path = analysis_dir / f"exp{int(exp_id):02d}_insight.md"
            _write_exp_insight_md(
                exp_id=exp_id,
                exp_df=exp_df,
                doc_df=doc_df,
                field_df=field_df,
                insight_path=insight_path,
                retrieve_k=rk,
                context_k=ck,
                recall_k=rck,
                analysis_extra=analysis_extra,
            )
            print(f"Saved per-exp insight: {insight_path}")

    summary_df = pd.concat(exp_rows, ignore_index=True)
    summary_path = out_dir / "exp_compare_explevel.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    if args.run_analysis and len(all_summaries) > 0:
        if len(all_fields) >= 2:
            pd.concat(all_fields, ignore_index=True).to_csv(
                analysis_dir / "field_breakdown_compare.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if len(all_hits) >= 2:
            pd.concat(all_hits, ignore_index=True).to_csv(
                analysis_dir / "hit_rank_dist_compare.csv",
                index=False,
                encoding="utf-8-sig",
            )

        md_path = analysis_dir / args.analysis_md_name
        lines: List[str] = []
        lines.append("# Failure Analysis Summary")
        lines.append("")
        lines.append(f"- exp_ids: {args.exp_ids}")
        lines.append(f"- retrieve_k: {rk}")
        lines.append(f"- context_k: {ck}")
        lines.append(f"- recall_k: {rck}")
        lines.append(f"- sim_threshold: {args.sim_threshold}")
        if args.n_docs:
            lines.append(f"- n_docs(limit): {args.n_docs}")
        lines.append("")
        for s in all_summaries:
            lines.append(f"## exp {s['exp_id']}")
            lines.append("")
            lines.append(f"- total rows: {s['n']}")
            lines.append(f"- fail rows (match=0): {s['fail']}")
            lines.append("")
            lines.append("### tag counts")
            lines.append("")
            lines.append("| tag | count |")
            lines.append("| --- | --- |")
            for k, v in s["tag_counts"].items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
            lines.append("### fail by field (top10)")
            lines.append("")
            lines.append("| field | count |")
            lines.append("| --- | --- |")
            for k, v in s["fail_by_field"].items():
                lines.append(f"| {k} | {int(v)} |")
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved analysis markdown: {md_path}")


if __name__ == "__main__":
    main()
