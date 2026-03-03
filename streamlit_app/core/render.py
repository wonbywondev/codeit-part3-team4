from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional

import fitz  # pymupdf


def get_pdf_num_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return int(n)


def render_pdf_page_png(pdf_path: str, page_1indexed: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if page_1indexed < 1 or page_1indexed > n:
        doc.close()
        raise ValueError(f"page out of range: {page_1indexed} (valid: 1..{n})")
    page = doc.load_page(page_1indexed - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    b = pix.tobytes("png")
    doc.close()
    return b


def _normalize_for_search(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_page_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_rects(rects: List[fitz.Rect]) -> List[fitz.Rect]:
    out: List[fitz.Rect] = []
    seen = set()
    for r in rects:
        key = (round(float(r.x0), 1), round(float(r.y0), 1), round(float(r.x1), 1), round(float(r.y1), 1))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _hangul_spaced_variant(s: str) -> str:
    compact = re.sub(r"\s+", "", s or "")
    if not compact:
        return ""
    # "사업명" -> "사 업 명" 형태 보조 매칭
    if re.fullmatch(r"[가-힣A-Za-z0-9&/()\-_.:]+", compact):
        return " ".join(list(compact))
    return ""


def _search_rects_fuzzy(page, query: str, max_hits_per_query: int = 25) -> List[fitz.Rect]:
    qn = _normalize_for_search(query)
    if not qn:
        return []

    rects = page.search_for(qn)
    if rects:
        return rects[:max_hits_per_query]

    # 긴 구절은 앞부분으로 보조 탐색
    if len(qn) > 40:
        rects = page.search_for(qn[:40])
        if rects:
            return rects[:max_hits_per_query]

    # 공백 제거 버전/글자 사이 공백 버전 탐색
    compact = re.sub(r"\s+", "", qn)
    if compact and compact != qn:
        rects = page.search_for(compact)
        if rects:
            return rects[:max_hits_per_query]
    spaced = _hangul_spaced_variant(qn)
    if spaced:
        rects = page.search_for(spaced)
        if rects:
            return rects[:max_hits_per_query]

    # 최종 폴백: 핵심 토큰 부분매칭 (글자분리/약간의 OCR 흔들림 대응)
    toks = [t for t in re.findall(r"[가-힣A-Za-z0-9&]+", qn) if len(t) >= 2]
    fallback_rects: List[fitz.Rect] = []
    for tok in toks[:6]:
        rs = page.search_for(tok)
        if not rs:
            tok_sp = _hangul_spaced_variant(tok)
            if tok_sp:
                rs = page.search_for(tok_sp)
        if rs:
            fallback_rects.extend(rs[:4])
    if fallback_rects:
        return _dedupe_rects(fallback_rects)[:max_hits_per_query]
    return []


def render_pdf_page_png_with_highlights(
    pdf_path: str,
    page_1indexed: int,
    highlight_queries: List[str],
    zoom: float = 2.0,
    max_hits_per_query: int = 25,
) -> Tuple[bytes, int]:
    """
    특정 페이지에서 highlight_queries를 찾아 하이라이트 주석을 올리고 PNG로 렌더.
    반환: (png_bytes, total_hits)
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if page_1indexed < 1 or page_1indexed > n:
        doc.close()
        raise ValueError(f"page out of range: {page_1indexed} (valid: 1..{n})")

    page = doc.load_page(page_1indexed - 1)

    total_hits = 0
    for q in highlight_queries:
        rects = _search_rects_fuzzy(page, q, max_hits_per_query=max_hits_per_query)
        if rects:
            rects = rects[:max_hits_per_query]
            total_hits += len(rects)
            for r in rects:
                annot = page.add_highlight_annot(r)
                annot.update()

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, total_hits


def find_pages_for_queries(
    pdf_path: str,
    queries: List[str],
    *,
    start_page: int = 1,
    max_pages_scan: int = 200,
    page_whitelist: Optional[List[int]] = None,
) -> List[int]:
    """
    PDF를 스캔해서 queries 중 하나라도 hit가 있는 페이지 목록(1-indexed)을 반환.
    start_page: 스캔 시작 물리적 페이지(1-indexed). 목차 등 앞 페이지를 제외할 때 사용.
    """
    if not queries:
        return []

    doc = fitz.open(pdf_path)
    n = doc.page_count
    p0_start = max(0, start_page - 1)
    limit = min(n, p0_start + max_pages_scan)
    if page_whitelist:
        scan_p0 = sorted(
            {
                p - 1 for p in page_whitelist
                if isinstance(p, int) and 1 <= p <= n and p >= start_page
            }
        )
    else:
        scan_p0 = list(range(p0_start, limit))

    pages: List[int] = []
    for p0 in scan_p0:
        page = doc.load_page(p0)
        page_text = _normalize_page_text(page.get_text("text"))
        hit_any = False
        for q in queries:
            qn = _normalize_for_search(q)
            if not qn:
                continue
            # 텍스트 레이어 미포함 페이지는 기존 search_for로만 판단
            if page_text and qn not in page_text:
                continue
            rects = page.search_for(qn)
            if not rects and len(qn) > 40:
                rects = page.search_for(qn[:40])
            if rects:
                hit_any = True
                break
        if hit_any:
            pages.append(p0 + 1)

    doc.close()
    return pages


def find_pages_with_hit_counts(
    pdf_path: str,
    queries: List[str],
    *,
    start_page: int = 1,
    max_pages_scan: int = 200,
    max_hits_per_query: int = 100,
    page_whitelist: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    PDF를 스캔해서 queries hit 개수를 페이지별로 합산해 반환.
    반환: {page_1indexed: total_hits}
    - start_page: 스캔 시작 물리적 페이지(1-indexed). 목차 등 앞 페이지를 제외할 때 사용.
    - 여러 query의 hit를 합산
    - hit가 0인 페이지는 포함하지 않음
    """
    if not queries:
        return {}

    doc = fitz.open(pdf_path)
    n = doc.page_count
    p0_start = max(0, start_page - 1)
    limit = min(n, p0_start + max_pages_scan)
    if page_whitelist:
        scan_p0 = sorted(
            {
                p - 1 for p in page_whitelist
                if isinstance(p, int) and 1 <= p <= n and p >= start_page
            }
        )
    else:
        scan_p0 = list(range(p0_start, limit))

    hit_map: Dict[int, int] = {}
    for p0 in scan_p0:
        page = doc.load_page(p0)
        page_text = _normalize_page_text(page.get_text("text"))
        total = 0
        for q in queries:
            qn = _normalize_for_search(q)
            if not qn:
                continue
            if page_text and qn not in page_text:
                continue
            rects = page.search_for(qn)
            if not rects and len(qn) > 40:
                rects = page.search_for(qn[:40])
            if rects:
                total += min(len(rects), max_hits_per_query)
        if total > 0:
            hit_map[p0 + 1] = total

    doc.close()
    return hit_map
