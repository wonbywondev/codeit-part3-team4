## 컬럼(지표) 하나씩 설명
1. doc_id
- 평가 대상 문서 단위
  - 각 PDF 하나가 하나의 RAG 실험 단위

2. n_questions_total
- 해당 문서에 대해 평가한 전체 질문 수
  - 지금은 모든 문서가 21문항으로 동일

3. n_questions_with_evidence
- gold evidence(정답 근거 페이지)가 정의된 질문 수
  - 21이면 → 모든 질문에 정답 근거가 있음
  - 평가셋 품질은 매우 좋음

4. retrieval_recall@k
- “정답 근거가 Top-K 검색 결과 안에 한 번이라도 들어왔는가?”
- coverage 지표
  - 1.0 = 모든 질문에서 정답 페이지를 최소 1번은 찾음
  - 예
    - 1.000 → 절대 놓치지 않음
    - 0.333 → 3문제 중 1문제만 찾음

5. retrieval_mrr@k
- “정답 근거가 얼마나 위에 랭크됐는가?”
  - ranking 품질 지표
  - 값이 높을수록 정답 chunk가 Top 1~2 근처
  - 예
    - 0.54 → 평균적으로 꽤 위
    - 0.09 → 거의 맨 아래

6. n_questions_with_gold_fields
- 실제로 정답 값(gold answer) 이 존재하는 질문 수
- 지금은 모두 21 → generation 평가 가능

7. gen_fill_rate
- “모델이 아무 말이라도 답을 한 비율”
  - 1.0 → 항상 응답함
  - 0.52 → 절반 정도는 “명시 없음 / 빈 값”
  - retrieval 문제가 아니라 generation/프롬프트 문제 신호일 수 있음

8. gen_match_rate
- “모델 답변이 gold 정답과 정확히 일치한 비율”
  - 가장 엄격한 지표
  - 0.23 = 21문항 중 약 5문항 정확히 맞춤

9. gen_avg_similarity
- “틀렸더라도 얼마나 의미적으로 가까운가”
  - 문자열 유사도 / 의미 유사도 기반
  - 30~35 이상이면 “사람이 보면 거의 맞다” 수준
  - 16~18이면 문맥은 비슷한데 핵심 누락

10. score
- 종합 점수
  - score = 0.5 × retrieval_mrr + 0.2 × retrieval_recall + 0.3 × gen_match_rate

  ### 문서별 핵심 인사이트
1. 1위 그룹 (가장 잘 되는 문서)
- (사)한국대학스포츠협의회 KUSF
  - retrieval_recall@k: 0.76
  - retrieval_mrr@k: 0.54
  - gen_match_rate: 0.24
  - score: 0.495 (전체 1위)
  - 해석
    - 구조가 명확한 문서
    - 질문 ↔ 근거 ↔ 답변 연결이 자연스러움

- 부산국제영화제 BIFF
  - recall 1.0 (절대 안 놓침)
  - MRR 0.5
  - gen_match_rate는 약간 낮음
  - 해석
    - 검색은 완벽
    - 하지만 표현이 다양해서 exact match가 어려움
    - 유사도 기반 평가가 중요함을 보여주는 케이스

2. 중간 그룹 (검색은 되지만 generation이 약함)
- 예: 예술경영지원센터 / 경희대
  - recall 0.76~0.9
  - MRR 중간
  - gen_similarity는 매우 높음 (30+)
  - 해석
    - 모델이 “무슨 말인지는 아는 상태”
    - 하지만 필드 단위 추출 정확한 값 포맷 어려움
    - 프롬프트 / 출력 포맷 개선으로 바로 점수 올라갈 문서들

3. 하위 그룹 (근본적으로 어려운 문서)
- 구미 아시아육상 / 벤처기업협회
  - recall 0.33~0.47
  - MRR 0.11대
  - gen_match 거의 없음
  - 해석
    - 문서 구조가 복잡하거나 정답이 표/서식/여러 페이지에 분산
    - chunk 전략 / Docling / 구조 기반 파싱 필요