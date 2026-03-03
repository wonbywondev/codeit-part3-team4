from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd

REQ_QUESTION_COLS = ["instance_id","qid","doc_id","question","type"]
REQ_EVID_COLS = ["instance_id","doc_id","page_start","page_end","anchor_text"]

def _norm_doc_id(doc_id: str) -> str:
    s = str(doc_id).strip().strip('"').strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _ensure_csv(path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8-sig")

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def next_xid_for_doc(doc_id: str, questions_base: Path, questions_user: Path) -> str:
    """
    doc_id별 qid(X###) 최대값 + 1
    - base+user 모두 스캔하여 충돌 방지
    - 기본 시작은 X011
    """
    doc_id = _norm_doc_id(doc_id)
    max_n = 10
    for p in [questions_base, questions_user]:
        if p.exists():
            df = pd.read_csv(p)
            if "doc_id" in df.columns and "qid" in df.columns:
                sub = df[df["doc_id"].astype(str).apply(_norm_doc_id) == doc_id]
                for qid in sub["qid"].astype(str).tolist():
                    m = re.match(r"X(\d+)", qid.strip())
                    if m:
                        max_n = max(max_n, int(m.group(1)))
    return f"X{max_n+1:03d}"

def make_instance_id(doc_id: str, qid: str) -> str:
    """
    기존 D001 같은 prefix가 doc_id에 있으면 사용하고,
    없으면 파일 stem 기반으로 U_<stem>_<qid>
    """
    doc_id_n = _norm_doc_id(doc_id)
    m = re.search(r"(D\d{3})", doc_id_n)
    prefix = m.group(1) if m else Path(doc_id_n).stem[:20]
    return f"U_{prefix}_{qid}"

def save_question_row(path_user: Path, row: dict):
    _ensure_csv(path_user, REQ_QUESTION_COLS)
    df = _read_csv_safe(path_user)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path_user, index=False, encoding="utf-8-sig")

def save_evidence_rows(path_user: Path, rows: list[dict]):
    _ensure_csv(path_user, REQ_EVID_COLS)
    df = _read_csv_safe(path_user)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path_user, index=False, encoding="utf-8-sig")

def append_gold_fields_jsonl(path_user: Path, record: dict):
    path_user.parent.mkdir(parents=True, exist_ok=True)
    with open(path_user, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")