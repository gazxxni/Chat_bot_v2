# 챗봇 RAG 성능 최적화 메모리

## 문제 상황
- `main_hybrid_rerank.py` (고도화 버전): 46% → 14% → 5% (성능 악화)
- `back.py` (이전 버전): 46-48% (안정적)
- **원인**: Query Expansion의 역효과 + 높은 임계값 + 불균형한 가중치

## 현재 개선 사항 (v1)

### 1. Gemini → OpenAI 교체 ✓
- 무료 계획 토큰 한도 문제 해결
- gpt-4o-mini 사용 중

### 2. 파라미터 최적화 ✓
| 파라미터 | 이전 | 개선 | 이유 |
|---------|------|------|------|
| BM25 가중치 | 60% | 70% | 한국어 정확 매칭 중요 |
| Vector 가중치 | 40% | 30% | 의미론적 유사도 보조 |
| 임계값 | 0.05 | 0.02 | 너무 높아서 답변 거부 |
| Context | 10000자 | 8000자 | LLM 집중력 향상 |
| Hybrid top_k | 30 | 40 | 후보 수 증가 |
| Rerank top_k | 12 | 15 | 최종 후보 수 증가 |
| Query Expansion | 활성 | 선택적 | 역효과 분석 가능 |

### 3. 프롬프트 개선 ✓
- "참고 정보에만 있는 과목" 강조
- 더 명확한 지시사항 추가
- 마크다운 기호 명시적 금지

## 다음 테스트 단계

### 테스트 1: 현재 개선사항 검증
```bash
# 터미널 1: main_hybrid_rerank.py 실행 (OpenAI)
uvicorn main_hybrid_rerank:app --port 8000

# 터미널 2: back.py 실행
uvicorn back:app --port 8001

# 터미널 3: 평가
python eval_rag_accuracy.py
```

### 테스트 2: Query Expansion 효과 측정
- `ENABLE_QUERY_EXPANSION = True` 로 변경 후 테스트
- 확장 쿼리가 정말 역효과인지 확인

### 테스트 3: 파라미터 세밀 조정 (if 성능 부족)
- BM25 가중치: 0.65, 0.75, 0.8 테스트
- Reranking 임계값: 0.01, 0.03, 0.04 테스트
- Context 길이: 7000, 9000 테스트

## 성능 지표
- **목표**: Back.py 48% recall 이상 달성
- **평가 기준**:
  - 유효 답변만 평가 (429 에러 제외)
  - 평균 키워드 리콜 (recall)
  - 완전 일치 정확도

## 추가 고려사항
1. **BM25 토큰화**: 현재 정규식 기반 한국어 토큰화 사용
   - 더 정교한 형태소 분석 필요시 konlpy 추가
2. **임베딩 모델**: text-embedding-3-small 사용
   - 더 큰 모델(3-large) 테스트 가능
3. **Cross-Encoder 모델**: mmarco-mMiniLMv2-L12-H384-v1 (다국어 지원)
   - 변경 필요시 confirm
