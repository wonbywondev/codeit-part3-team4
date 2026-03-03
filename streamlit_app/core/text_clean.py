from __future__ import annotations

import re
import unicodedata
from typing import List

_TOKEN_RE = re.compile(r"\S+")

# ✅ 반복문자 4회 이상이면 1개로 축약 (예: 222222 -> 2, ㅋㅋㅋㅋ -> ㅋ, ---- -> -)
_REPEAT_CHAR_4PLUS = re.compile(r"(.)\1{3,}")  # same char repeated >= 4


def normalize_unicode(text: str) -> str:
    # 전각(｢ ｣ 등) / 이상한 공백 등을 정규화
    return unicodedata.normalize("NFKC", text)


def collapse_consecutive_duplicate_tokens(text: str, keep: int = 1) -> str:
    """
    '2024 2024 2024' / '년 년 년' 처럼 공백 기반 연속 토큰 중복 축약.
    """
    toks = _TOKEN_RE.findall(text)
    if not toks:
        return text
    out: List[str] = []
    prev = None
    run = 0
    for t in toks:
        if t == prev:
            run += 1
        else:
            prev = t
            run = 1
        if run <= keep:
            out.append(t)
    return " ".join(out)


def squash_repeated_chars_4plus(text: str) -> str:
    """
    PDF 겹침 등으로 생기는 반복문자(4회 이상)를 1개로 축약.
    예: '2222222' -> '2', 'ㅋㅋㅋㅋ' -> 'ㅋ', '----' -> '-'
    """
    if not text:
        return ""
    return _REPEAT_CHAR_4PLUS.sub(r"\1", text)


def post_clean_text(text: str) -> str:
    """
    pp_v6 산출물/추출 텍스트에 대해 '앱 레벨'에서 안전하게 노이즈 완화.
    - NFKC 정규화(전각 제거)
    - 과도한 공백/빈줄 정리
    - 과도한 특수문자 반복 축약
    - ✅ 반복문자(4회 이상) 축약 (서비스에서도 222222/ㅋㅋㅋㅋ 제거)
    - 연속 중복 토큰 축약(가장 중요)
    """
    if not text:
        return ""

    t = text.replace("\x00", " ")
    t = normalize_unicode(t)

    # 공백/개행 정리
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{4,}", "\n\n", t)

    # 점선/기호 반복 축약(너무 긴 점묘류)
    t = re.sub(r"[·.…]{5,}", " ", t)

    # ✅ 핵심: 반복문자 4회+ → 1개
    t = squash_repeated_chars_4plus(t)

    # 핵심: 연속 토큰 중복 축약
    t = collapse_consecutive_duplicate_tokens(t, keep=1)

    return t.strip()