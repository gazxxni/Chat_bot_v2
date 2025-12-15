import re
import pdfplumber
from pathlib import Path

PDF_PATH = Path("역량중심 교육과정 개발 보고서_컴퓨터공학과_축약.pdf")

def count_semantic_blocks(pdf_path: Path):
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

    print(f"분석 완료")
    print(f"총 추출된 데이터 블록 수: {len(blocks)} 개")
    
    print("-" * 30)
    print("앞부분 3개 블록 예시:")
    for i, (p, t, c) in enumerate(blocks[:3]):
        print(f"[{i+1}] 페이지: {p} | 제목: {t} | 내용 길이: {len(c)}자")
    print("-" * 30)

if __name__ == "__main__":
    if PDF_PATH.exists():
        count_semantic_blocks(PDF_PATH)
    else:
        print(f"파일을 찾을 수 없습니다: {PDF_PATH}")