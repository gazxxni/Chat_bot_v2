# main_hybrid_rerank.py - Query Expansion + Hybrid Search + Re-ranking (Gemini API)
# 필요 설치: pip install rank-bm25 sentence-transformers google-generativeai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import google.generativeai as genai
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── Gemini API 설정 ────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("[INFO] Gemini API 설정 완료 (gemini-2.5-flash)")

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

# ─── Query Expansion (Gemini) ───────────────────────────
def expand_query(query: str) -> str:
    """
    사용자 질문을 확장하여 관련 키워드 추가
    예: "인공지능 관련 과목" → "인공지능 머신러닝 딥러닝 알고리즘 자료구조 파이썬 선형대수"
    """
    prompt = f"""
    당신은 컴퓨터공학과 교육과정 검색을 돕는 전문가입니다.
    
    학생의 질문: "{query}"
    
    이 질문으로 교육과정 문서를 검색할 때, 관련된 과목이나 키워드를 최대한 많이 찾을 수 있도록 검색어를 확장해주세요.
    
    규칙:
    1. 원래 질문의 핵심 키워드는 반드시 포함
    2. 해당 분야의 기초/선수 과목 키워드 추가 (예: 인공지능 → 알고리즘, 자료구조, 선형대수, 확률통계)
    3. 관련 심화 과목 키워드 추가 (예: 인공지능 → 머신러닝, 딥러닝, 텍스트마이닝)
    4. 관련 프로그래밍 언어 추가 (예: 파이썬, C언어)
    5. 띄어쓰기로 구분된 키워드만 출력 (문장 형태 X)
    6. 최대 15개 키워드
    
    확장된 검색어만 출력하세요:
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        expanded = response.text.strip()
        print(f"[DEBUG] Query Expansion: '{query}' → '{expanded}'")
        return expanded
    except Exception as e:
        print(f"[ERROR] Query Expansion 실패: {e}")
        return query

# ─── Hybrid Search ──────────────────────────────────────
def hybrid_search(query: str, top_k: int = 30):
    """BM25 60% + Vector 40%"""
    results = {}
    bm25_weight = 0.6
    vector_weight = 0.4
    
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
    
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    print(f"[DEBUG] Re-ranking 상위 3개:")
    for i, (doc, sc) in enumerate(scored[:3]):
        print(f"  #{i+1} (score={sc:.2f}): {doc[:50]}...")
    
    return [doc for doc, _ in scored[:top_k]]

# ─── /ask 엔드포인트 ───────────────────────────────────
@app.post("/ask")
async def ask_question(request: QueryRequest):
    original_query = request.query.strip()
    print(f"\n{'='*50}\n[질문] {original_query}\n{'='*50}")
    
    # Step 1: Query Expansion (Gemini)
    expanded_query = expand_query(original_query)
    
    # Step 2: Hybrid Search (확장된 쿼리로 검색)
    hybrid_res = hybrid_search(expanded_query, top_k=30)
    candidates = [r[1]["document"] for r in hybrid_res]
    
    if not candidates:
        return {"answer": "관련 정보를 찾을 수 없습니다."}
    
    print(f"[INFO] Hybrid Search 결과: {len(candidates)}개")
    
    # Step 3: Re-ranking (원래 질문 기준)
    reranked = rerank_documents(original_query, candidates, top_k=12)
    
    # Step 4: 컨텍스트 구성
    context = "\n\n---\n\n".join(list(dict.fromkeys(reranked)))[:4500]
    print(f"[INFO] 컨텍스트: {len(context)}자")
    
    # Step 5: Gemini로 응답 생성
    final_prompt = f"""
    당신은 컴퓨터공학과 신입생의 과목 선택을 도와주는 조교 챗봇입니다.
    상담 선생님처럼 부드럽고 친근하게 설명해주세요.
    
    [필수 규칙]
    1. 참고 정보에 있는 과목 중 질문과 관련된 과목은 모두 추천하세요.
    2. 관련 과목이 많으면 많이, 적으면 적게 추천하세요. 개수 제한 없음.
    3. 기초 과목 → 핵심 과목 → 심화 과목 순서로 추천하세요.
    4. 각 과목은 "과목명 (학년-학기): 한 줄 설명" 형식으로 작성하세요.
    5. 참고 정보에 없는 과목은 절대 언급하지 마세요.
    
    [금지 사항]
    - *, **, -, •, # 같은 마크다운 기호 사용 금지
    - 번호는 "1. 2. 3." 형식만 사용
    
    [참고 정보]
    {context}
    
    [학생의 질문]
    {original_query}
    
    위 참고 정보에서 질문과 관련된 모든 과목을 찾아 추천해주세요.
    """

    try:
        response = gemini_model.generate_content(final_prompt)
        answer = response.text
        
        return {
            "answer": answer,
            "debug": {
                "original_query": original_query,
                "expanded_query": expanded_query,
                "candidates_count": len(candidates),
                "reranked_count": len(reranked)
            }
        }
    except Exception as e:
        print(f"[ERROR] Gemini 응답 생성 실패: {e}")
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
        "message": "컴공도우미봇 (Query Expansion + Hybrid + Re-ranking + Gemini)",
        "docs": len(ALL_DOCUMENTS),
        "model": "gemini-2.5-flash"
    }
