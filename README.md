# 컴퓨터공학과 교과과정 안내 챗봇의 디벨롭 버전

컴퓨터공학과 신입생 및 재학생의 성공적인 수강 계획을 돕기 위해 설계된 RAG(검색 증강 생성) 기반 챗봇 프로젝트입니다.

## 1. Key Improvements: From Research to Advanced RAG

본 프로젝트는 기존 챗봇 프로젝트를 바탕으로, **실제 서비스 레벨의 검색 정확도와 응답 속도를 확보하기 위해 아키텍처를 대폭 고도화**하였습니다. 단순한 벡터 검색을 넘어 **Hybrid Search, Re-ranking, Semantic Chunking** 등 Advanced RAG 기술을 적용했습니다.

## 2. Technical Evolution: Basic vs. Production Code

| 구분 | 기존 챗봇 (Baseline) | 실제 구현 코드 (Advanced RAG) | 개선 효과 |
| :--- | :--- | :--- | :--- |
| **Retrieval** | **Dense Retrieval**<br>(Vector Similarity 100%) | **Hybrid Search**<br>(BM25 60% + Vector 40%) | '운영체제', '학점' 등 고유명사 키워드 검색 성능 대폭 개선 |
| **Ranking** | **Top-K Selection**<br>(유사도 순 상위 10개 단순 추출) | **2-Stage Re-ranking**<br>(Cross-Encoder 기반 정밀 재순위화) | 문맥 연관성을 심층 분석하여 Hallucination 감소 및 답변 품질 극대화 |
| **Query** | **Raw Query**<br>(사용자 질문 그대로 검색) | **Query Expansion**<br>(LLM을 활용한 검색어 확장) | "인공지능" 검색 시 "머신러닝, 딥러닝"까지 포괄하여 Recall(재현율) 상승 |
| **Model** | **Gemini 1.5 Pro** | **Gemini 2.5 Flash** | 응답 지연(Latency) 최소화 및 운영 비용 효율화 |

## 3. Detailed Technical Implementations

#### 1. Retrieval: 하이브리드 검색 (Hybrid Search)
- **구현:** `rank_bm25` (Keyword Search)와 `ChromaDB` (Vector Search)를 결합했습니다.
- **전략:** 테스트 결과, 학과 정보 특성상 정확한 용어 매칭이 중요하여 **BM25(0.6) : Vector(0.4)**의 가중치를 적용했습니다. 이를 통해 의미적 유사성뿐만 아니라 사용자가 의도한 정확한 키워드가 포함된 문서를 우선적으로 확보했습니다.

#### 2. Ranking: 2-Stage Re-ranking
- **1단계 (Retrieval):** 하이브리드 검색을 통해 후보 문서(Candidates) **30개**를 빠르게 추출합니다.
- **2단계 (Re-ranking):** 다국어 처리가 가능한 Cross-Encoder 모델(`mmarco-mMiniLMv2`)을 사용하여, 사용자 질문과 후보 문서 간의 연관성을 정밀 채점합니다.
- **최종 선별:** 점수가 가장 높은 상위 **12개** 문서만 LLM의 컨텍스트로 제공하여, 토큰 비용을 절약하고 환각(Hallucination) 가능성을 차단했습니다.

#### 3. Performance & Reliability
- **Latency 개선:** 논문 단계의 `Gemini 1.5 Pro` 대신, 최신 경량화 모델인 `Gemini 2.5 Flash`를 도입하여 답변 생성 속도를 개선했습니다.
- **Guardrails:** 시스템 프롬프트에 "참고 정보에 없는 과목은 절대 언급하지 말 것" 등의 제약 조건을 명시하여 답변의 신뢰성을 확보했습니다.

---

#### 초기 버전 & 상세 설명
[![GitHub](https://img.shields.io/badge/GitHub-View_Repository-181717?style=for-the-badge&logo=github)](https://github.com/gazxxni/Chat_bot)
