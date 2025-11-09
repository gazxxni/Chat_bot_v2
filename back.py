# 필요한 라이브러리들을 가져옵니다.
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from openai import OpenAI  # ⭐️ Google 대신 OpenAI를 가져옵니다.

# ─── 환경변수 로드 ──────────────────────────────────────
load_dotenv()
# ⭐️ OpenAI 키만 불러오도록 수정합니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── OpenAI 클라이언트 설정 ────────────────────────────────
# ⭐️ OpenAI API 키를 사용하여 클라이언트를 초기화합니다.
client = OpenAI(api_key=OPENAI_API_KEY)

# ─── ChromaDB 클라이언트 및 컬렉션 초기화 ────────────────
# ChromaDB 설정은 text-embedding-3-small을 이미 사용하고 있으므로 그대로 둡니다.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection_semantic = chroma_client.get_or_create_collection(
    name="semantic_education_chunks",
    embedding_function=embedding_fn
)

# ─── FastAPI 앱 생성 ────────────────────────────────────
app = FastAPI()

# ─── CORS 설정 (React 프론트 허용) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 요청 바디 모델 정의 ────────────────────────────────
class QueryRequest(BaseModel):
    query: str

# ─── 질문 처리 엔드포인트 ───────────────────────────────
@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query.strip()
    top_k = 10

    # ChromaDB에서 관련 문서 검색
    try:
        results_semantic = collection_semantic.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        return {"answer": f"DB 검색 중 오류가 발생했습니다: {str(e)}"}

    retrieved_docs = []
    if results_semantic.get("documents"):
        retrieved_docs.extend(results_semantic["documents"][0])

    if not retrieved_docs:
        retrieved_docs = ["관련된 정보를 찾을 수 없습니다."]

    # 검색된 문서 정제 (중복 제거, 3개 제한, 1500자 제한)
    retrieved_docs = list(dict.fromkeys(retrieved_docs))[:10]
    context = "\n".join(retrieved_docs)[:1500]

    # ⭐️ OpenAI GPT 모델에 최적화된 프롬프트 구성 (시스템/사용자 분리)
    
    # 1. 시스템 프롬프트: 챗봇의 역할과 말투 정의
    system_prompt = """
    당신은 컴퓨터공학과 신입생들의 과목 선택을 도와주는 조교 챗봇입니다.
    학생들은 과목 순서, 트랙 구성, 진로와 관련된 수업을 고민하고 있으며, 
    당신은 이들에게 마치 상담 선생님처럼 부드럽고 명확하게 설명해 줍니다.
    기술 용어(예: 위상정렬, 최단경로)는 사용하지 말고, 
    자연스럽고 따뜻한 말투로 설명해주세요. 
    "먼저 ~을 듣고, 그 다음 ~을 듣는 게 좋아요" 같은 말투를 사용해 주세요. 
    중요한 과목은 강조해도 좋아요. 
    학생들이 헷갈리지 않도록 순서를 정리해서 말해 주세요.
    주의: 답변에는 `*`, `**`, `-`, `•` 등과 같은 특수 기호를 사용하지 말아 주세요. 
    필요한 경우 부드러운 문장으로 자연스럽게 강조해주세요.
    """

    # 2. 사용자 프롬프트: 실제 맥락(context)과 질문(query) 전달
    user_prompt = f"""
    다음은 참고 정보입니다:
    {context}

    학생의 질문:
    {query}
    """

    # ⭐️ Gemini 호출 대신 OpenAI GPT 모델 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo", "gpt-4o-mini" 등
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        # ⭐️ 응답 구조가 다르므로 .text 대신 .choices[0].message.content 사용
        return {"answer": response.choices[0].message.content}
    
    except Exception as e:
        # ⭐️ 오류 메시지 변경
        return {"answer": f"OpenAI 응답 생성 중 오류가 발생했습니다: {str(e)}"}

# ─── 기본 루트 테스트 ──────────────────────────────────
@app.get("/")
def root():
    return {"message": "컴공도우미봇 API가 실행 중입니다 🚀"}