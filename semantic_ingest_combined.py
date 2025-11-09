# semantic_ingest_combined.py – 의미 기반 청크 생성 및 저장 통합 스크립트
# 컬렉션 명: semantic_education_chunks
import os
import re
import uuid
import logging
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# ─── 환경변수 로딩 ──────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 필요합니다.")

# ─── 설정 ──────────────────────────────────────────────
PDF_PATH = Path("역량중심 교육과정 개발 보고서_컴퓨터공학과_축약.pdf")
DB_PATH = "./chroma_db"
COL_NAME = "semantic_education_chunks"    # 컬렉션 명
OUTPUT_TEXT_FILE = "semantic_blocks_output_best.txt"

# ─── 로깅 설정 ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 의미 블록 추출 함수 ───────────────────────────────
def extract_semantic_blocks(pdf_path: Path):
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            current_block = []
            current_title = None

            for line in lines:
                if re.match(r"^(교과목명|역량명|교과개요|전공역량|<표 \d+>|[가-힣]+\.)", line):
                    if current_block:
                        blocks.append((page_num, current_title or "일반", "\n".join(current_block)))
                        current_block = []
                    current_title = line
                else:
                    current_block.append(line)

            if current_block:
                blocks.append((page_num, current_title or "일반", "\n".join(current_block)))

    logger.info(f"{len(blocks)}개의 의미 블록 추출됨")
    return blocks

# ─── ChromaDB 저장 함수 ────────────────────────────────
def store_blocks_to_chroma(blocks):
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"  # 최신 임베딩 모델 적용  text-embedding-3-large  text-embedding-3-small
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    if COL_NAME in client.list_collections():
        logger.info(f"기존 컬렉션 '{COL_NAME}' 삭제")
        client.delete_collection(name=COL_NAME)

    col = client.get_or_create_collection(name=COL_NAME, embedding_function=embedding_fn)

    for idx, (page, title, content) in enumerate(blocks):
        col.add(
            ids=[str(uuid.uuid4())],
            documents=[content],
            metadatas=[{
                "page": page,
                "section_title": title,
                "source": "역량중심 교육과정 보고서"
            }]
        )

    logger.info(f"{len(blocks)}개 블록이 ChromaDB에 저장됨")

# ─── 블록을 txt 파일로 저장하는 함수 ─────────────────────
def save_blocks_to_txt(blocks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (page, title, content) in enumerate(blocks, start=1):
            f.write(f"==== 블록 {idx} ====\n")
            f.write(f"페이지 번호: {page}\n")
            f.write(f"섹션 제목: {title}\n")
            f.write(f"내용:\n{content}\n")
            f.write("\n\n")
    logger.info(f"블록 데이터를 '{output_path}' 파일로 저장 완료")

# ─── 메인 실행부 ───────────────────────────────────────
if __name__ == "__main__":
    blocks = extract_semantic_blocks(PDF_PATH)
    store_blocks_to_chroma(blocks)
    save_blocks_to_txt(blocks, OUTPUT_TEXT_FILE)
    print("semantic_education_chunks 임베딩 저장 완료")

