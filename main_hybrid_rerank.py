# main_hybrid_rerank.py - Query Expansion + Hybrid Search + Re-ranking (OpenAI API)
# 필요 설치: pip install rank-bm25 sentence-transformers openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
import re
import math

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── OpenAI 클라이언트 설정 ────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
print("[INFO] OpenAI API 설정 완료 (gpt-4o-mini)")

# ─── Re-ranker 모델 로드 ─────────────────────────────────
print("[INFO] Cross-Encoder 모델 로딩 중...")
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
print("[INFO] Cross-Encoder 모델 로딩 완료 (다국어 지원)")

# ─── ChromaDB 초기화 ────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

existing_collections = chroma_client.list_collections()
print(f"[INFO] 존재하는 컬렉션: {existing_collections}")

collection_semantic = chroma_client.get_or_create_collection(
    name="semantic_education_chunks",
    embedding_function=embedding_fn
)

# ─── BM25 인덱스 구축 ─────────────────────────────────────
def load_all_documents():
    all_data = collection_semantic.get()
    documents = all_data.get("documents", [])
    ids = all_data.get("ids", [])
    print(f"[DEBUG] 로드된 문서 수: {len(documents)}")
    return documents, ids

def tokenize_korean(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

ALL_DOCUMENTS, ALL_IDS = load_all_documents()
TOKENIZED_DOCS = [tokenize_korean(doc) for doc in ALL_DOCUMENTS]
BM25_INDEX = BM25Okapi(TOKENIZED_DOCS) if TOKENIZED_DOCS else None
print(f"[INFO] BM25 인덱스: {len(ALL_DOCUMENTS)}개 문서")

# ─── FastAPI ────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ─── 설정 ──────────────────────────────────
ENABLE_QUERY_EXPANSION = True  # Query Expansion 활성화 (Q22 해결)

# ─── Query Expansion (OpenAI) ───────────────────────────
def expand_query(query: str) -> str:
    """
    사용자 질문을 확장하여 관련 키워드 추가
    예: "인공지능 관련 과목" → "인공지능 머신러닝 딥러닝 알고리즘 자료구조 파이썬 선형대수"
    """
    system_prompt = """당신은 컴퓨터공학과 교육과정 검색을 돕는 전문가입니다.
    학생의 질문에서 검색어를 확장할 때:
    1. 원래 질문의 핵심 키워드는 반드시 포함
    2. 해당 분야의 기초/선수 과목 키워드 추가 (예: 인공지능 → 알고리즘, 자료구조, 선형대수, 확률통계)
    3. 관련 심화 과목 키워드 추가 (예: 인공지능 → 머신러닝, 딥러닝, 텍스트마이닝)
    4. 관련 프로그래밍 언어 추가 (예: 파이썬, C언어)
    5. 띄어쓰기로 구분된 키워드만 출력 (문장 형태 X)
    6. 최대 15개 키워드
    """

    user_prompt = f"""질문: "{query}"

위 질문으로 교육과정 문서를 검색할 때, 관련된 과목이나 키워드를 최대한 많이 찾을 수 있도록 검색어를 확장해주세요.
확장된 검색어만 출력하세요:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        expanded = response.choices[0].message.content.strip()
        print(f"[DEBUG] Query Expansion: '{query}' → '{expanded}'")
        return expanded
    except Exception as e:
        print(f"[ERROR] Query Expansion 실패: {e}")
        return query

# ─── Hybrid Search ──────────────────────────────────────
def hybrid_search(query: str, top_k: int = 70):
    """BM25 85% + Vector 15% (정확 매칭 극대화)"""
    results = {}
    bm25_weight = 0.85
    vector_weight = 0.15
    
    # BM25
    if BM25_INDEX:
        tokens = tokenize_korean(query)
        scores = BM25_INDEX.get_scores(tokens)
        for idx, score in enumerate(scores):
            if score > 0:
                results[ALL_IDS[idx]] = {
                    "document": ALL_DOCUMENTS[idx],
                    "bm25": score,
                    "vector": 0.0
                }
        print(f"[DEBUG] BM25 매칭: {sum(1 for s in scores if s > 0)}개")
    
    # Vector
    try:
        vr = collection_semantic.query(query_texts=[query], n_results=top_k, include=["documents", "distances"])
        if vr.get("ids") and vr["ids"][0]:
            print(f"[DEBUG] Vector 매칭: {len(vr['ids'][0])}개")
            for i, doc_id in enumerate(vr["ids"][0]):
                sim = 1 - vr["distances"][0][i]
                if doc_id in results:
                    results[doc_id]["vector"] = sim
                else:
                    results[doc_id] = {"document": vr["documents"][0][i], "bm25": 0.0, "vector": sim}
    except Exception as e:
        print(f"[ERROR] Vector Search: {e}")
    
    # 점수 결합
    if results:
        max_bm25 = max(r["bm25"] for r in results.values()) or 1
        max_vec = max(r["vector"] for r in results.values()) or 1
        for r in results.values():
            r["score"] = (bm25_weight * r["bm25"] / max_bm25) + (vector_weight * r["vector"] / max_vec)
    
    sorted_res = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    return sorted_res[:top_k]

# ─── Re-ranking ─────────────────────────────────────────
def rerank_documents(query: str, documents: list, top_k: int = 10):
    if not documents:
        return []
    
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    probs = [sigmoid(s) for s in scores]
    
    scored = sorted(zip(documents, probs, scores), key=lambda x: x[1], reverse=True)
    
    print(f"[DEBUG] Re-ranking 상위 3개 (확률 / 로짓):")
    for i, (doc, prob, logit) in enumerate(scored[:3]):
        print(f"  #{i+1}: {prob:.4f} (logit={logit:.2f}) - {doc[:30]}...")
    
    return [(doc, prob) for doc, prob, _ in scored[:top_k]]

# ─── /ask 엔드포인트 ───────────────────────────────────
@app.post("/ask")
async def ask_question(request: QueryRequest):
    original_query = request.query.strip()
    print(f"\n{'='*50}\n[질문] {original_query}\n{'='*50}")

    # Step 1: Query Expansion (선택사항)
    if ENABLE_QUERY_EXPANSION:
        expanded_query = expand_query(original_query)
    else:
        expanded_query = original_query
        print(f"[INFO] Query Expansion 비활성화 - 원본 쿼리 사용")

    # Step 2: Hybrid Search (쿼리로 검색)
    hybrid_res = hybrid_search(expanded_query, top_k=70)

    # Hybrid 단계에서는 너무 엄격하게 자르지 않음 (혹시 모르니)
    if not hybrid_res:
        return {"answer": "관련 정보를 찾을 수 없습니다."}

    candidates = [r[1]["document"] for r in hybrid_res]
    print(f"[INFO] Hybrid Search 결과: {len(candidates)}개")

    # Step 3: Re-ranking (원래 질문 기준)
    reranked_with_scores = rerank_documents(original_query, candidates, top_k=30)
    
    if not reranked_with_scores:
        return {"answer": "관련 정보를 찾을 수 없습니다."}

    top_doc, top_prob = reranked_with_scores[0]
    print(f"[DEBUG] 가장 높은 관련도 점수: {top_prob:.4f}")

    if top_prob < 0.005:
        print("[INFO] 관련성 매우 낮음 -> 답변 거부")
        return {"answer": "질문하신 내용과 관련된 학과 정보를 찾을 수 없습니다."}
    
    valid_docs = [doc for doc, prob in reranked_with_scores]
    
    if not valid_docs:
         return {"answer": "관련 정보를 찾을 수 없습니다."}
     
    # Step 4: 컨텍스트 구성
    context = "\n\n---\n\n".join(list(dict.fromkeys(valid_docs)))[:5500]
    print(f"[INFO] 컨텍스트: {len(context)}자")

    # Step 5: OpenAI로 응답 생성
    system_prompt = """당신은 컴퓨터공학과 신입생의 과목 선택을 도와주는 조교 챗봇입니다.
상담 선생님처럼 부드럽고 친근하게 설명해주세요.

[필수 규칙 - 반드시 준수하세요. 위반하면 안됨]
1. **절대 규칙: 참고 정보에만 있는 내용**을 사용하세요.
   - 참고 정보에 없는 과목/정보는 절대 절대 만들지 마세요.
   - 없는 학점을 말하지 마세요 (예: "123학점" 같은 거짓)
   - 없는 학기를 말하지 마세요 (예: "5학년" 같은 거짓)
2. **절대 규칙: 관련된 모든 항목을 포함**하세요.
   - 누락된 것이 없는지 참고 정보를 처음부터 끝까지 다시 꼼꼼히 확인하세요.
   - 절반만 답변하면 안됩니다.
3. **기초 → 중급 → 심화** 순서로 정렬하세요.
4. 각 항목은 "이름/내용 (학년-학기): 설명" 형식으로 작성하세요.
5. 정보가 없으면 "질문하신 내용과 관련된 정보를 찾을 수 없습니다"라고 명확히 말하세요.

[금지 사항 - 이것을 위반하면 안됨]
- 마크다운 기호(*,**,-,•,#) 절대 절대 금지
- 번호는 "1. 2. 3." 형식만 허용
- 참고 정보에 없는 사실/숫자를 절대 만들지 마세요
- 거짓 정보 생성 금지 (이 규칙이 가장 중요함)

[검증 체크리스트 - 답변 전에 확인]
- 모든 항목을 참고 정보와 하나하나 대조했나요?
- 누락된 관련 항목이 정말 없나요? (중요!)
- 거짓 정보(만든 정보)가 정말 없나요? (매우 중요!)
- 참고 정보의 모든 항목을 다 포함했나요?"""

    user_prompt = f"""[참고 정보]
{context}

[학생의 질문]
{original_query}

========================================
[매우 중요한 주의사항]
========================================
1. **반드시** 위의 "참고 정보"에 있는 내용만 사용하세요.
2. **절대** 참고 정보에 없는 과목, 학점, 학기를 만들지 마세요.
3. **관련된 모든 항목**을 빠짐없이 포함하세요 (절반만 답변하면 안됨).
4. 거짓 정보를 생성하면 안됩니다 (예: 없는 학점을 말하기).
5. 답변하기 전에 참고 정보의 모든 항목을 확인하세요.

[특히 주의: 다음을 확인하세요]
- 참고 정보의 모든 과목을 다 언급했나요?
- 만든 정보(거짓)가 없나요?
- 참고 정보에 없는 내용을 말하지 않았나요?"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "debug": {
                "original_query": original_query,
                "expanded_query": expanded_query,
                "candidates_count": len(candidates),
                "reranked_count": len(reranked_with_scores)
            }
        }
    except Exception as e:
        print(f"[ERROR] OpenAI 응답 생성 실패: {e}")
        return {"answer": f"오류: {e}"}

# ─── 디버깅 엔드포인트 ────────────────────────────────
@app.get("/debug")
def debug_info():
    return {
        "documents": len(ALL_DOCUMENTS),
        "bm25_ready": BM25_INDEX is not None,
        "collections": chroma_client.list_collections()
    }

@app.get("/search_keyword/{keyword}")
def search_keyword(keyword: str):
    matched = []
    for i, doc in enumerate(ALL_DOCUMENTS):
        if keyword in doc:
            matched.append({"index": i, "preview": doc[:400]})
    return {"keyword": keyword, "count": len(matched), "docs": matched[:15]}

@app.post("/test_expansion")
async def test_expansion(request: QueryRequest):
    """Query Expansion만 테스트"""
    query = request.query.strip()
    expanded = expand_query(query)
    return {"original": query, "expanded": expanded}

@app.post("/test_full_pipeline")
async def test_full_pipeline(request: QueryRequest):
    """전체 파이프라인 테스트 (응답 생성 제외)"""
    original = request.query.strip()
    expanded = expand_query(original)
    
    hybrid_res = hybrid_search(expanded, top_k=20)
    candidates = [r[1]["document"] for r in hybrid_res]
    
    reranked = rerank_documents(original, candidates, top_k=10)
    
    return {
        "original_query": original,
        "expanded_query": expanded,
        "hybrid_count": len(candidates),
        "reranked_preview": [doc[:200] for doc in reranked[:5]]
    }

@app.get("/")
def root():
    return {
        "message": "컴공도우미봇 (Query Expansion + Hybrid + Re-ranking + OpenAI)",
        "docs": len(ALL_DOCUMENTS),
        "model": "gpt-4o-mini"
    }
