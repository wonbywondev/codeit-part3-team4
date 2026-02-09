# codeit-part3-team4

### 새로 만들어야 할 폴더/파일 목록
1. 코드(모듈) 위치
```
src/
  eval/
    __init__.py
    retrieval_metrics.py
    generation_metrics.py
    dataset.py
```

2. 실행 스크립트 위치
```
scripts/
  run_retrieval_eval.py
  run_extraction_eval.py
  run_qa_eval.py
  run_recommend_eval.py   # (L4, 선택)
```

3 평가 데이터 위치
```
data/
  eval/
    gold/
      questions.csv   # (이미 있음)
      gold_evidence.csv
      gold_fields.jsonl
      company_profiles.jsonl        # (L4용, 선택)
      labels_recommend.jsonl        # (L4용, 선택)
    runs/
      ... (자동 생성)
```
### L1~L4 평가를 “실제로 돌리는” 최소 요구사항
1. L1 Retrieval (필수)
- 필요 파일
  - questions.csv
  - gold_evidence.csv ❗(직접 작성 필요: “정답 페이지”)
- 필요 코드
  - retrieval_metrics.py + run_retrieval_eval.py

2. L2 Extraction
- 필요 파일
  - gold_fields.jsonl ❗(직접 작성 필요: “정답 필드 값”)
- 필요 코드
  - generation_metrics.py + run_extraction_eval.py

3. L3 QA/요약
- 필요 파일
  - QA는 gold를 “정답 문장”으로 만들기 어려우니 근거 인용 정확도 + 근거 일치율(근거 밖 말하면 0) 중심으로 평가 가능
  - 최소로는 gold_evidence.csv만 있어도 가능
- 필요 코드
  - run_qa_eval.py (답변 생성 + 근거 비교)

4. L4 추천/적합도/확률/금액
- 필요 파일(없으면 평가 불가)
  - company_profiles.jsonl (고객사 프로필)
  - labels_recommend.jsonl (적합/부적합 또는 점수 라벨)
- 필요 코드
  - run_recommend_eval.py

- 현실적으로 팀 프로젝트에서는 L1 + L2 + L3까지만 제대로 해도 충분히 강함.
- L4는 “확장 기능”으로 설계/시연 중심으로 발표하면 좋고, 라벨이 있으면 정량까지.