# streamlit_app/core/llm.py
from __future__ import annotations

from typing import Any, List
from openai import OpenAI


def _supports_temperature(model: str) -> bool:
    m = (model or "").strip().lower()
    if m.startswith("gpt-5"):
        return False
    if m.startswith("o"):
        return False
    return True


def _extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = getattr(resp, "output", None) or []
    collected: List[str] = []
    for item in out:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) == "output_text":
                t = getattr(part, "text", None)
                if t:
                    collected.append(str(t))
    return "\n".join([c.strip() for c in collected if c.strip()]).strip()


def summarize_with_evidence(
    api_key: str,
    model: str,
    query: str,
    evidence: str,
    pages: list[int],
    chat_history: str = "",
    memory_facts: str = "",
    *,
    temperature: float = 0.2,
    max_completion_tokens: int = 512,
    fallback_model: str = "gpt-4.1-mini",
    max_retries: int = 2,
    reasoning_effort: str = "low",
) -> str:
    client = OpenAI(api_key=api_key)

    system = (
        "너는 근거 기반 요약/질의응답 도우미다. "
        "반드시 제공된 Evidence를 우선 근거로 답하라. "
        "Evidence가 부족하면 [최근 대화 요약]/[기확인 사실]을 보조근거로만 사용할 수 있다. "
        "근거가 없으면 'NOT_FOUND'라고 말해라. "
        "반드시 마지막 줄에 '근거 페이지: p.xx, p.yy' 형식으로 페이지를 출력해라. "
        "페이지가 없으면 '근거 페이지: NOT_FOUND'로 출력해라."
    )

    user = f"""
[질문/요청]
{query}

[최근 대화 요약]
{chat_history or "(없음)"}

[기확인 사실]
{memory_facts or "(없음)"}

[Evidence]
{evidence}

[근거 페이지 후보]
{pages}
""".strip()

    def _one_call(use_model: str, effort: str) -> str:
        kwargs = dict(
            model=use_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=max_completion_tokens,
        )

        m = (use_model or "").lower()
        if m.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": effort, "summary": "auto"}

        if _supports_temperature(use_model):
            kwargs["temperature"] = temperature

        resp = client.responses.create(**kwargs)
        return _extract_text(resp)

    text = _one_call(model, reasoning_effort)

    tries = 0
    while (not text) and tries < max_retries:
        tries += 1
        text = _one_call(model, reasoning_effort)

    if not text and fallback_model:
        text = _one_call(fallback_model, "medium")

    if not text:
        raise RuntimeError("LLM이 텍스트 메시지를 생성하지 못했습니다.")

    # 마지막 줄 근거 페이지 강제 보정
    lines = [ln.rstrip() for ln in text.splitlines()]
    if not any(ln.strip().startswith("근거 페이지:") for ln in lines[-3:]):
        if pages:
            pages_s = ", ".join([f"p.{int(p)}" for p in sorted(set(int(x) for x in pages if int(x) > 0))])
        else:
            pages_s = "NOT_FOUND"
        lines.append(f"근거 페이지: {pages_s}")
        text = "\n".join(lines).strip()

    return text
