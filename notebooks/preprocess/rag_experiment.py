# preprocess/rag_experiment.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gc
import json
import re
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


# -------------------------
# Config / Prompt
# -------------------------
CONFIG = {
    "chunk_length": 800,          # C1 baseline
    "top_k": 15,
    "retrieve_k": 24,             # retrieval eval candidate size
    "context_k": 8,               # generation context subset size
    "max_tokens": 2000,           # non gpt-5
    "max_completion_tokens": 2000,  # gpt-5
    "temperature": 0.1,           # non gpt-5
    "alpha": 0.7,                 # hybrid weight for vector score
    "max_context_chars": 4000,    # context hard cap (chars)
    "target_field_max_context_chars": 2200,
}



# JSON-only prompt (태그 방식 제거)

# RFP_PROMPT = """역할: 너는 RFP/입찰 공고 문서(CONTEXT 발췌)에서 정보를 추출한다.

# 절대 규칙:
# 1) 근거는 CONTEXT에 있는 문자열만 사용한다(추측 금지).
# 2) 출력은 JSON 객체 1개만. 코드블록/설명/추가 텍스트 금지.
# 3) 키는 QUESTIONS의 key를 정확히 그대로 사용한다(키 추가/삭제/변경 금지).
# 4) 값은 모두 string으로 출력한다.
# 5) CONTEXT에 명확한 근거가 있을 때만 채워라. 애매하거나 유사한 것 같아도 확신할 수 없으면 NOT_FOUND를 써라.
# 6) "NOT_FOUND"는 정말로 근거가 전혀 없을 때만 사용한다.
# 7) 날짜는 문서에 나온 형식 그대로 사용한다. 조사/어미 붙이지 말 것. (예: "~까지", "~부터" 금지)
# 8) 금액은 문서에 나온 형식 그대로 사용한다. 단위(원)가 문서에 있으면 붙이고, 없으면 붙이지 말 것. (예: "100,000,000원" 또는 "100,000,000")
# 9) 값은 최대한 짧고 핵심만. 단, 아래 12~14번 규칙이 있는 필드는 해당 규칙을 우선 적용하라.
# 10) 라벨/항목명은 값에 포함하지 말 것. (예: "발주기관 : 국민연금공단" → "국민연금공단", "사업기간 : 6개월" → "6개월")
# 11) 금액 표기 시 괄호 안 부연설명 붙이지 말 것. 숫자와 단위만. (예: "50,000,000(금 오천만원/VAT포함)" → "50,000,000원")
# 12) requirements_must, eligibility는 문서에 나열된 항목을 " / "로 구분해서 나열하라. 임의로 요약하거나 생략하지 말 것. (예: "SW사업자 등록 / 최근 3년 실적 1건 이상 / 정보보안관리체계 인증")
# 13) eval_items는 "항목명:배점" 형식으로 " / "로 구분해서 나열하라. (예: "기술평가:90 / 가격평가:10")
# 14) purpose는 문서의 사업목적 문장을 그대로 발췌하되 2문장 이내로 한정하라. contract_type은 계약 방식 명칭만 출력하라. (예: "협상에 의한 계약", "제한경쟁입찰")


# 작업 방법(반드시 따름):
# 먼저 CONTEXT에서 다음 유형의 신호를 찾아라: 사업명/용역명, 금액(원), 기간(일/개월), 기관명, 마감일, 평가(기술/가격), 요구사항/자격/평가항목/계약방식/사업목적.
# 찾은 신호가 있으면 해당 key에 매핑해 값을 채워라.
# 확실한 매핑이 불가능하면 NOT_FOUND.

# QUESTIONS(JSON array):
# {questions_json}

# CONTEXT:
# {context}
# """.strip()


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
    """
    returns [(type, question), ...]
    - doc_id == "*" rows are common questions
    - doc_id == doc_name rows are per-doc questions
    - if 'type' duplicates exist, keep last (per-doc overrides common)
    """
    common = questions_df[questions_df["doc_id"] == "*"][["type", "question"]]
    per_doc = questions_df[questions_df["doc_id"] == doc_name][["type", "question"]]
    merged = pd.concat([common, per_doc], ignore_index=True)

    merged["type"] = merged["type"].astype(str)
    merged["question"] = merged["question"].astype(str)

    # 중복 type이 있으면 뒤(per_doc) 우선
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


def find_hit_rank(chunks: List[str], idxs: List[int], anchors: List[str]) -> Optional[int]:
    for rank, ci in enumerate(idxs, start=1):
        if 0 <= int(ci) < len(chunks):
            c = chunks[int(ci)]
            if any(a in c for a in anchors):
                return rank
    return None


_NOT_FOUND_TOKENS = {"", "없음", "notfound", "not_found", "gen_fail"}
_MONEY_KEY_HINTS = ("budget", "amount", "cost", "price")
_DATE_KEY_HINTS = ("deadline", "date", "start", "end")
_NUMBER_KEY_HINTS = (
    "count",
    "ratio",
    "rate",
    "time",
    "year",
    "users",
    "score",
    "threshold",
    "limit",
    "day",
    "month",
    "hour",
    "retention",
    "period",
    "duration",
)
_MONEY_MULTIPLIER = {
    "원": 1.0,
    "krw": 1.0,
    "천원": 1_000.0,
    "만원": 10_000.0,
    "백만원": 1_000_000.0,
    "천만원": 10_000_000.0,
    "억원": 100_000_000.0,
    "천": 1_000.0,
    "만": 10_000.0,
    "백만": 1_000_000.0,
    "천만": 10_000_000.0,
    "억": 100_000_000.0,
}


def _to_float(num_s: str) -> Optional[float]:
    try:
        return float(num_s.replace(",", "").strip())
    except Exception:
        return None


def _is_not_found(text: str) -> bool:
    return str(text or "").strip().lower() in _NOT_FOUND_TOKENS


def _infer_eval_mode(field: Optional[str]) -> str:
    key = str(field or "").lower()
    if any(h in key for h in _MONEY_KEY_HINTS):
        return "money"
    if any(h in key for h in _DATE_KEY_HINTS):
        return "date"
    if any(h in key for h in _NUMBER_KEY_HINTS):
        return "number"
    return "text"


def _extract_money_won(text: str) -> Optional[int]:
    s = str(text or "")
    for m in re.finditer(
        r"(\d[\d,]*(?:\.\d+)?)\s*(억원|천만원|백만원|만원|천원|억|천만|백만|만|천|원|krw)",
        s,
        flags=re.IGNORECASE,
    ):
        base = _to_float(m.group(1))
        unit = (m.group(2) or "").lower()
        if base is None or unit not in _MONEY_MULTIPLIER:
            continue
        return int(round(base * _MONEY_MULTIPLIER[unit]))

    # 금액 표기가 단순 숫자인 경우(예: 90000000) fallback
    nums = _extract_numbers(s)
    if len(nums) == 1:
        return int(round(nums[0]))
    return None


def _extract_dates(text: str) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    s = str(text or "")
    for m in re.finditer(r"(20\d{2})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})", s):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.add((y, mo, d))
    return out


def _extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    s = str(text or "")
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", s):
        n = _to_float(token)
        if n is not None:
            out.append(n)
    return out


def _has_exact_numeric_overlap(pred_nums: List[float], gold_nums: List[float], tol: float = 1e-9) -> bool:
    for p in pred_nums:
        for g in gold_nums:
            if abs(p - g) <= tol:
                return True
    return False


def eval_gen(
    pred: str,
    gold: Optional[str],
    threshold: int = 80,
    field: Optional[str] = None,
) -> Dict[str, float]:
    pred = (pred or "").strip()

    # 기존 정의 유지(원하면 NOT_FOUND 제외로 바꿀 수 있음)
    fill = 1.0 if pred and pred.lower() not in _NOT_FOUND_TOKENS else 0.0

    if gold is None or str(gold).strip() == "":
        return {
            "fill": fill,
            "match": np.nan,
            "sim": np.nan,
            "strict_match": np.nan,
            "strict_applied": 0.0,
        }

    gold = str(gold).strip()
    mode = _infer_eval_mode(field)

    if mode == "money":
        pred_money = _extract_money_won(pred)
        gold_money = _extract_money_won(gold)
        if pred_money is not None and gold_money is not None:
            is_match = float(pred_money == gold_money)
            return {
                "fill": fill,
                "match": is_match,
                "sim": 100.0 if is_match else 0.0,
                "strict_match": is_match,
                "strict_applied": 1.0,
            }

    if mode == "date":
        pred_dates = _extract_dates(pred)
        gold_dates = _extract_dates(gold)
        if pred_dates and gold_dates:
            is_match = float(bool(pred_dates & gold_dates))
            return {
                "fill": fill,
                "match": is_match,
                "sim": 100.0 if is_match else 0.0,
                "strict_match": is_match,
                "strict_applied": 1.0,
            }

    if mode == "number":
        pred_nums = _extract_numbers(pred)
        gold_nums = _extract_numbers(gold)
        if pred_nums and gold_nums:
            is_match = float(_has_exact_numeric_overlap(pred_nums, gold_nums))
            return {
                "fill": fill,
                "match": is_match,
                "sim": 100.0 if is_match else 0.0,
                "strict_match": is_match,
                "strict_applied": 1.0,
            }

    sim = fuzz.token_set_ratio(pred, gold)
    return {
        "fill": fill,
        "match": 1.0 if sim >= threshold else 0.0,
        "sim": float(sim),
        "strict_match": np.nan,
        "strict_applied": 0.0,
    }


def _nanmean_safe(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float(np.nan)
    return float(np.nanmean(arr))


def _build_context_from_indices(chunks: List[str], idxs: List[int], max_chars: Optional[int] = None) -> str:
    parts: List[str] = []
    for rank, ci in enumerate(idxs, start=1):
        if 0 <= int(ci) < len(chunks):
            parts.append(f"[CHUNK {rank}]\n{chunks[int(ci)]}")
    context = "\n\n".join(parts)
    if max_chars is not None and max_chars > 0:
        return context[:max_chars]
    return context


_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[\-\*\•·▪◦□■◆▶▷☞]+|\(?\d+\)?[.)]?|[가-힣A-Za-z][.)])\s*(.+)$"
)


def _extract_list_items(context: str) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()

    for raw in str(context or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("[CHUNK "):
            continue
        m = _LIST_PREFIX_RE.match(line)
        if not m:
            continue
        item = m.group(1).strip(" -:;")
        if len(item) < 2:
            continue
        if len(item) > 220:
            item = item[:220]
        if item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _is_count_question(question: str) -> bool:
    q = re.sub(r"\s+", "", str(question or ""))
    return bool(re.search(r"(총몇|몇개|개수|갯수|수량|몇가지|몇종)", q))


def _extract_ordinal_index(question: str) -> Optional[int]:
    q = re.sub(r"\s+", "", str(question or ""))
    m = re.search(r"(\d+)(?:번째|번(?:째)?)", q)
    if m:
        return int(m.group(1))

    word_map = {
        "첫번째": 1,
        "첫째": 1,
        "두번째": 2,
        "둘째": 2,
        "세번째": 3,
        "셋째": 3,
        "네번째": 4,
        "넷째": 4,
        "다섯번째": 5,
        "여섯번째": 6,
        "일곱번째": 7,
        "여덟번째": 8,
        "아홉번째": 9,
        "열번째": 10,
    }
    for token, idx in word_map.items():
        if token in q:
            return idx
    return None


def _rule_based_list_answer(question: str, context: str) -> Optional[str]:
    n = _extract_ordinal_index(question)
    need_count = _is_count_question(question)
    if n is None and not need_count:
        return None

    items = _extract_list_items(context)
    if not items:
        return None

    if need_count:
        return str(len(items))

    if n is not None:
        if 1 <= n <= len(items):
            return items[n - 1]
        return "NOT_FOUND"
    return None


_TARGET_TUNED_FIELDS = {
    "requirements_must",
    "eligibility",
    "eval_items",
    "purpose",
    "contract_type",
}

_FIELD_CONTEXT_HINTS: Dict[str, List[str]] = {
    "requirements_must": ["요구사항", "필수", "준수", "의무", "반드시", "필요", "must"],
    "eligibility": ["입찰참가자격", "참가자격", "자격", "요건", "조건", "실적", "면허"],
    "eval_items": ["평가항목", "배점", "점수", "기술평가", "가격평가", "평가기준"],
    "purpose": ["사업목적", "목적", "배경", "추진", "필요성"],
    "contract_type": ["계약방법", "입찰방식", "계약형태", "협상에 의한 계약", "제한경쟁", "일반경쟁"],
}

_NOISY_PRED_HINTS = {"검토항목", "검토의견", "세부 내용", "산출정보", "평가기준", "배점"}


def _is_target_tuned_field(field: Optional[str]) -> bool:
    return str(field or "").strip().lower() in _TARGET_TUNED_FIELDS


def _clean_pred_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_noisy_pred(text: str) -> bool:
    s = _clean_pred_text(text)
    if not s:
        return True
    if s.count("|") >= 2:
        return True
    if len(s) > 220:
        return True
    return any(tok in s for tok in _NOISY_PRED_HINTS)


def _filter_context_for_field(field: Optional[str], context: str) -> str:
    f = str(field or "").strip().lower()
    if f not in _FIELD_CONTEXT_HINTS:
        return context

    lines = str(context or "").splitlines()
    hints = _FIELD_CONTEXT_HINTS[f]
    matched: set[int] = set()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[CHUNK "):
            continue
        if _LIST_PREFIX_RE.match(line):
            matched.add(i)
        if any(h in line for h in hints):
            matched.add(i)
            if i - 1 >= 0:
                matched.add(i - 1)
            if i + 1 < len(lines):
                matched.add(i + 1)

    if not matched:
        return context

    kept = [lines[i] for i in sorted(matched)]
    reduced = "\n".join(kept).strip()
    if not reduced:
        return context

    max_chars = int(CONFIG.get("target_field_max_context_chars", 2200) or 2200)
    return reduced[:max_chars]


def _extract_contract_type(text: str) -> Optional[str]:
    s = str(text or "")
    patterns = [
        ("제한경쟁입찰", r"제한\s*경쟁\s*입찰"),
        ("일반경쟁입찰", r"일반\s*경쟁\s*입찰"),
        ("협상에 의한 계약", r"협상\s*에\s*의한\s*계약"),
        ("수의계약", r"수의\s*계약"),
    ]
    found: List[str] = []
    for label, pat in patterns:
        if re.search(pat, s):
            found.append(label)
    if not found:
        return None
    return ", ".join(found)


def _extract_purpose_from_context(context: str) -> Optional[str]:
    for raw in str(context or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("[CHUNK "):
            continue
        m = re.search(r"(?:목적|배경|추진)\s*[:：]\s*(.+)$", line)
        if m:
            v = m.group(1).strip(" -|")
            if v:
                return v[:160]
        if any(tok in line for tok in ["사업목적", "추진배경", "추진 목적"]):
            return line[:160]
    return None


def _extract_list_like_from_context(field: str, context: str, max_items: int = 5) -> Optional[str]:
    field_hints = _FIELD_CONTEXT_HINTS.get(field, [])
    items: List[str] = []
    for raw in str(context or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("[CHUNK "):
            continue
        if "|" in line:
            continue
        is_list = bool(_LIST_PREFIX_RE.match(line))
        has_hint = any(h in line for h in field_hints)
        if not is_list and not has_hint:
            continue
        line = re.sub(r"^\s*(?:[\-\*\•·▪◦□■◆▶▷☞]+|\(?\d+\)?[.)]?|[가-힣A-Za-z][.)])\s*", "", line)
        line = line.strip(" -:;")
        if len(line) < 3:
            continue
        if len(line) > 120:
            line = line[:120]
        if line not in items:
            items.append(line)
        if len(items) >= max_items:
            break
    if not items:
        return None
    return "; ".join(items)


def _postprocess_target_field_pred(field: Optional[str], pred: str, context: str) -> str:
    f = str(field or "").strip().lower()
    p = _clean_pred_text(pred)

    if f == "contract_type":
        ct = _extract_contract_type(p) or _extract_contract_type(context)
        return ct if ct else (p if p else "NOT_FOUND")

    if f == "purpose":
        if _is_not_found(p) or _is_noisy_pred(p):
            alt = _extract_purpose_from_context(context)
            return alt if alt else ("NOT_FOUND" if _is_not_found(p) else p)
        return p[:180]

    if f in {"requirements_must", "eligibility", "eval_items"}:
        if _is_not_found(p) or _is_noisy_pred(p):
            alt = _extract_list_like_from_context(f, context, max_items=5)
            return alt if alt else ("NOT_FOUND" if _is_not_found(p) else p)
        return p

    return p if p else "NOT_FOUND"


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
        """
        queries: [(type_key, question_text), ...]
        returns: {type_key: answer_string}
        """
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
        chunks = _chunk_for_index(doc_path.name, size=CONFIG["chunk_length"])
        if chunks is None:
            text = clean_text(extract_text(doc_path))
            s = CONFIG["chunk_length"]
            return [text[i:i+s] for i in range(0, len(text), s)]
        return chunks


class C4DoclingChunker(BaseChunker):
    """Docling section_path 청킹을 강제 사용하고 싶을 때 선택"""
    def chunk(self, doc_path: Path) -> List[str]:
        chunks = _chunk_for_index(doc_path.name, size=CONFIG["chunk_length"])
        if chunks is None:
            return C1FixedChunker(size=CONFIG["chunk_length"]).chunk(doc_path)
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

class R4RerankerRetriever(BaseRetriever):
    def __init__(self, embed_model, reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.embed_model = embed_model
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(reranker_model, use_fp16=True)

    def build_index(self, chunks):
        from sentence_transformers import SentenceTransformer
        embs = self.embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs.astype("float32"))
        return index

    def retrieve(self, index, queries, top_k=20):
        q_embs = self.embed_model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
        q_mean = q_embs.mean(axis=0, keepdims=True).astype("float32")
        _, I = index.search(q_mean, min(top_k * 3, index.ntotal))
        candidates = I[0].tolist()
        return candidates[:top_k]

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

        # debug fields
        self.last_raw_text: str = ""
        self.last_resp_dump: Optional[Dict[str, Any]] = None
        self.last_debug: Dict[str, Any] = {}
    
    def generate(self, queries: List[Tuple[str, str]], context: str) -> Dict[str, str]:
        """
        Returns dict: {type_key: answer}
        Sentinel policy:
        - NOT_FOUND: 컨텍스트에 근거해 없다고 판단(정상 부재) 또는 key 자체 누락
        - GEN_FAIL: 모델 무응답/비정상(빈 출력, JSON 파싱 실패, key는 있는데 값이 공백/None 등)
        """
        NOT_FOUND = "NOT_FOUND"
        GEN_FAIL = "GEN_FAIL"

        MAX_CTX_CHARS = CONFIG.get("max_context_chars", 4000)
        context = (context or "")[:MAX_CTX_CHARS]

        q_payload = [{"key": k, "question": q} for k, q in queries]
        questions_json = json.dumps(q_payload, ensure_ascii=False)
        prompt = RFP_PROMPT.format(questions_json=questions_json, context=context)

        # debug init
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
                max_output_tokens=CONFIG.get("max_completion_tokens", 1000),
                reasoning={"effort": "minimal"},
            )
            self.last_resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else None
            self.last_debug["response_status"] = getattr(resp, "status", None)

            usage = getattr(resp, "usage", None)
            self.last_debug["output_tokens"] = getattr(usage, "output_tokens", None)

            text = (getattr(resp, "output_text", "") or "").strip()
            self.last_raw_text = text
            self.last_debug["output_text_repr"] = repr(text[:200])

            # 모델이 아예 무응답(빈 출력)
            if not text:
                return all_sentinel(GEN_FAIL)

            # JSON 파싱 실패 = 생성 실패로 간주
            try:
                obj = json.loads(text)
            except Exception as e:
                self.last_debug["parse_error"] = repr(e)
                return all_sentinel(GEN_FAIL)

            # JSON이 dict가 아니면 실패 취급
            if not isinstance(obj, dict):
                self.last_debug["parse_error"] = f"non-dict-json: {type(obj)}"
                return all_sentinel(GEN_FAIL)

            out: Dict[str, str] = {}
            for k, _q in queries:
                # key 자체가 없으면(모델이 누락) -> NOT_FOUND로 둬서 "부재"로 기록
                # (원하면 이것도 GEN_FAIL로 바꿀 수 있음)
                if k not in obj:
                    out[k] = NOT_FOUND
                    continue

                v_raw = obj.get(k)

                # key는 있는데 값이 None/공백이면 -> GEN_FAIL (모델이 답을 비워둔 것)
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
    # chunker
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

    # retriever
    if spec.retriever == "R1":
        retriever = R1BM25Retriever()
    elif spec.retriever == "R2":
        retriever = R2VectorRetriever(embed_model)
    elif spec.retriever == "R3":
        retriever = R3HybridRetriever(embed_model, bm25_candidates=200)
    elif spec.retriever == "R4":
        retriever = R4RerankerRetriever(embed_model)
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

        retrieve_k = int(CONFIG.get("retrieve_k", top_k) or top_k)
        retrieve_k = max(1, retrieve_k)
        context_k = int(CONFIG.get("context_k", min(8, retrieve_k)) or min(8, retrieve_k))
        context_k = max(1, min(context_k, retrieve_k))

        pred_map: Dict[str, str] = {}
        g_list: List[Dict[str, float]] = []
        r_list: List[Dict[str, float]] = []
        hit_ranks: List[float] = []
        field_rows: List[Dict[str, Any]] = []

        for field, question in queries:
            idxs_full = self.retriever.retrieve(index, [question], top_k=retrieve_k)
            idxs_ctx = idxs_full[:context_k]
            max_ctx = (
                CONFIG.get("max_context_chars_per_question")
                or CONFIG.get("max_context_chars")
            )
            context = _build_context_from_indices(chunks, idxs_ctx, max_chars=max_ctx)
            gen_context = _filter_context_for_field(field, context) if _is_target_tuned_field(field) else context

            # 요구사항 n번째/개수 질문은 리스트 라인 규칙 기반으로 우선 처리
            rule_pred = _rule_based_list_answer(question, gen_context)
            if rule_pred is not None:
                pred = str(rule_pred).strip()
            else:
                one_pred = self.generator.generate([(field, question)], gen_context)
                pred = (one_pred.get(field) or "").strip()
            if _is_target_tuned_field(field):
                pred = _postprocess_target_field_pred(field, pred, gen_context)
            pred_map[field] = pred

            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None
            g = eval_gen(pred, gold, threshold=sim_threshold, field=field)
            g_list.append(g)

            for _, row in qdf[qdf["field"].astype(str) == str(field)].iterrows():
                iid = str(row["instance_id"])
                anchors = GOLD_ANCHOR.get(iid, [])
                if anchors:
                    ret = eval_retrieval_by_anchor(chunks, idxs_full, anchors)
                    r_list.append(ret)
                    hr = find_hit_rank(chunks, idxs_full, anchors)
                    hit_ranks.append(float(hr) if hr is not None else np.nan)
                else:
                    r_list.append({"recall": np.nan, "mrr": np.nan})
                    hit_ranks.append(np.nan)

            field_rows.append(
                {
                    "field": str(field),
                    "question": str(question),
                    "pred": pred,
                    "gold": (str(gold) if gold is not None else ""),
                    "match": float(g["match"]) if np.isfinite(g["match"]) else np.nan,
                    "sim": float(g["sim"]) if np.isfinite(g["sim"]) else np.nan,
                    "strict_match": float(g["strict_match"]) if np.isfinite(g["strict_match"]) else np.nan,
                    "strict_applied": float(g["strict_applied"]),
                    "retrieve_k": int(retrieve_k),
                    "context_k": int(context_k),
                }
            )

        hit_arr = np.array(hit_ranks, dtype=float) if hit_ranks else np.array([], dtype=float)
        hit_r1 = float(np.nanmean((hit_arr == 1).astype(float))) if hit_arr.size else float(np.nan)
        hit_r2 = float(np.nanmean((hit_arr == 2).astype(float))) if hit_arr.size else float(np.nan)
        hit_r1_r2 = float(np.nanmean(((hit_arr == 1) | (hit_arr == 2)).astype(float))) if hit_arr.size else float(np.nan)

        metrics: Dict[str, Any] = {
            "doc_id": doc_name,
            "n_questions": int(len(qdf)),
            "chunk_count": int(len(chunks)),
            "retrieve_k": int(retrieve_k),
            "context_k": int(context_k),
            "pred_map": pred_map,
            "field_rows": field_rows,

            "ret_recall": _nanmean_safe([x["recall"] for x in r_list]),
            "ret_mrr": _nanmean_safe([x["mrr"] for x in r_list]),
            "hit_rank_mean": _nanmean_safe(hit_ranks),
            "hit_r1_ratio": hit_r1,
            "hit_r2_ratio": hit_r2,
            "hit_r1_r2_ratio": hit_r1_r2,

            "gen_fill": _nanmean_safe([x["fill"] for x in g_list]),
            "gen_match": _nanmean_safe([x["match"] for x in g_list]),
            "gen_sim": _nanmean_safe([x["sim"] for x in g_list]),
            "gen_match_strict": _nanmean_safe([x["strict_match"] for x in g_list]),
            "gen_strict_coverage": _nanmean_safe([x["strict_applied"] for x in g_list]),
        }

        del chunks, index, qdf, r_list, g_list, queries, GOLD_ANCHOR, pred_map, field_rows
        gc.collect()
        return metrics



    def run_single_doc_metrics(
        self,
        doc_path: Path,
        gold_fields_df: pd.DataFrame,
        gold_evidence_df: pd.DataFrame,
        top_k: int = 15,
        sim_threshold: int = 80,
        warn_on_mismatch: bool = True,
    ) -> Dict[str, Any]:
        doc_name = unicodedata.normalize("NFC", doc_path.name)

        # 1) 질문 로드: [(type, question), ...]
        queries = get_queries_for_doc(doc_name, self.questions_df)
        q_texts = [q for _t, q in queries]
        type_keys = [t for t, _q in queries]
        retrieve_k = int(CONFIG.get("retrieve_k", top_k) or top_k)
        retrieve_k = max(1, retrieve_k)
        context_k = int(CONFIG.get("context_k", min(8, retrieve_k)) or min(8, retrieve_k))
        context_k = max(1, min(context_k, retrieve_k))

        # 2) chunk -> index -> retrieve
        chunks = self.chunker.chunk(doc_path)
        index = self.retriever.build_index(chunks)
        idxs_full = self.retriever.retrieve(index, q_texts, top_k=retrieve_k)
        idxs_ctx = idxs_full[:context_k]

        # 3) context
        max_ctx = CONFIG.get("max_context_chars")
        context = _build_context_from_indices(chunks, idxs_ctx, max_chars=max_ctx)

        # 4) generate: returns dict {type: answer}
        pred_map = self.generator.generate(queries, context)

        # list answers in the same order as queries (for baseline-like eval)
        answers = [pred_map.get(t, "NOT_FOUND") for t in type_keys]

        expected_answer_count = len(q_texts)
        answer_count = len(answers)
        if warn_on_mismatch and answer_count != expected_answer_count:
            print(
                f"WARN answer_count mismatch | doc={doc_name} | "
                f"expected={expected_answer_count} got={answer_count}"
            )

        # --- evaluation ---
        qdf = gold_fields_df[gold_fields_df["doc_id"].astype(str) == doc_name].copy()
        GOLD_ANCHOR = build_gold_anchor_map(gold_evidence_df)

        # debug meta
        answers_preview = [str(x) for x in (answers[:5] if answers else [])]
        n_nonempty_answers = int(sum(1 for a in (answers or []) if str(a).strip()))
        n_notfound_answers = int(sum(1 for a in (answers or []) if str(a).strip().lower() in {"notfound", "not_found", "없음"}))

        raw_text = getattr(self.generator, "last_raw_text", None)
        raw_text_len = None if raw_text is None else int(len(str(raw_text).strip()))
        raw_text_preview = None if raw_text is None else str(raw_text)[:200]

        # 5) generation eval
        g_list: List[Dict[str, float]] = []
        preds: List[str] = []

        for i, (field, question) in enumerate(queries):
            gold_row = qdf[qdf["field"].astype(str) == str(field)]
            gold = gold_row["gold"].iloc[0] if not gold_row.empty else None

            pred = answers[i] if i < len(answers) else ""
            field_ctx = _filter_context_for_field(field, context) if _is_target_tuned_field(field) else context
            rule_pred = _rule_based_list_answer(question, field_ctx)
            if rule_pred is not None:
                pred = rule_pred
            pred_s = (pred or "").strip()
            if _is_target_tuned_field(field):
                pred_s = _postprocess_target_field_pred(field, pred_s, field_ctx)
            preds.append(pred_s)
            pred_map[field] = pred_s

            g = eval_gen(pred_s, gold, threshold=sim_threshold, field=field)
            g_list.append(g)

        pred_preview = preds[:5]
        n_nonempty_preds = int(sum(1 for p in preds if str(p).strip()))
        n_notfound_preds = int(sum(1 for p in preds if str(p).strip().lower() in {"notfound", "not_found", "없음"}))

        # 6) retrieval eval
        r_list: List[Dict[str, float]] = []
        hit_ranks: List[float] = []
        for _, row in qdf.iterrows():
            iid = str(row["instance_id"])
            anchors = GOLD_ANCHOR.get(iid, [])
            if anchors:
                ret = eval_retrieval_by_anchor(chunks, idxs_full, anchors)
                r_list.append(ret)
                hr = find_hit_rank(chunks, idxs_full, anchors)
                hit_ranks.append(float(hr) if hr is not None else np.nan)
            else:
                r_list.append({"recall": np.nan, "mrr": np.nan})
                hit_ranks.append(np.nan)

        hit_arr = np.array(hit_ranks, dtype=float) if hit_ranks else np.array([], dtype=float)
        hit_r1 = float(np.nanmean((hit_arr == 1).astype(float))) if hit_arr.size else float(np.nan)
        hit_r2 = float(np.nanmean((hit_arr == 2).astype(float))) if hit_arr.size else float(np.nan)
        hit_r1_r2 = float(np.nanmean(((hit_arr == 1) | (hit_arr == 2)).astype(float))) if hit_arr.size else float(np.nan)

        metrics: Dict[str, Any] = {
            "doc_id": doc_name,

            # debug/validation
            "expected_answer_count": int(expected_answer_count),
            "answer_count": int(answer_count),

            "n_questions": int(len(qdf)),
            "chunk_count": int(len(chunks)),
            "context_length": int(len(context)),
            "retrieve_k": int(retrieve_k),
            "context_k": int(context_k),

            # generator debug
            "raw_text_len": raw_text_len,
            "raw_text_preview": raw_text_preview,
            "answers_preview": answers_preview,
            "n_nonempty_answers": n_nonempty_answers,
            "n_notfound_answers": n_notfound_answers,
            "pred_preview": pred_preview,
            "n_nonempty_preds": n_nonempty_preds,
            "n_notfound_preds": n_notfound_preds,

            # NEW: json-style outputs
            "pred_map": pred_map,  # type->answer dict (원하면 저장/JSON dump에 바로 사용)

            "ret_recall": _nanmean_safe([x["recall"] for x in r_list]),
            "ret_mrr": _nanmean_safe([x["mrr"] for x in r_list]),
            "hit_rank_mean": _nanmean_safe(hit_ranks),
            "hit_r1_ratio": hit_r1,
            "hit_r2_ratio": hit_r2,
            "hit_r1_r2_ratio": hit_r1_r2,

            "gen_fill": _nanmean_safe([x["fill"] for x in g_list]),
            "gen_match": _nanmean_safe([x["match"] for x in g_list]),
            "gen_sim": _nanmean_safe([x["sim"] for x in g_list]),
            "gen_match_strict": _nanmean_safe([x["strict_match"] for x in g_list]),
            "gen_strict_coverage": _nanmean_safe([x["strict_applied"] for x in g_list]),
        }

        # cleanup
        del chunks, index, context, answers, qdf, r_list, g_list, idxs_full, idxs_ctx, queries, q_texts, GOLD_ANCHOR, preds, pred_map
        gc.collect()

        return metrics
