"""
PDF 解析与预处理：提取文本、清洗、结构化分块
"""
import re
import json
import pdfplumber
from pathlib import Path
from .config import PDF_PATH, PARSED_TEXT_PATH, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS

# ── 页脚模式（每页重复，需要移除）────────────────────────────
FOOTER_PATTERN = re.compile(
    r'\n\s*2026 Formula 1 Technical Regulations\s*\n'
    r'.*?\n'
    r'.*?F.d.ration Internationale.*?\n'
    r'\s*24 June 2024\s*Issue 8\s*',
    re.DOTALL
)

# ── Article / Section 边界检测 ──────────────────────────────────
ARTICLE_BOUNDARY = re.compile(r'^(ARTICLE\s+\d+)', re.MULTILINE)
SECTION_BOUNDARY = re.compile(r'^(\d+\.\d+(?:\.\d+)*)\s', re.MULTILINE)
APPENDIX_BOUNDARY = re.compile(r'^(APPENDIX\s+\d+)', re.MULTILINE)


def extract_text(pdf_path: Path) -> str:
    """用 pdfplumber 提取全文（跨平台兼容）"""
    all_pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_pages.append(text)
            if (i + 1) % 50 == 0:
                print(f"      Extracted page {i + 1}/{len(pdf.pages)}")
    return "\n\n".join(all_pages)


def clean_text(text: str) -> str:
    """移除页脚、修复常见噪声"""
    text = FOOTER_PATTERN.sub('\n', text)
    text = re.sub(r'�', "'", text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # 合并过多空行
    return text.strip()


def split_into_blocks(text: str) -> list[dict]:
    """
    按文章/附录边界 + 章节号层级切分为块
    返回: [{"content": "...", "article": "3", "section": "3.5", "title": "Floor Bodywork"}, ...]
    """
    raw_blocks = []
    # 在 ARTICLE / APPENDIX 边界处切分
    parts = ARTICLE_BOUNDARY.split(text)
    if not parts:
        parts = APPENDIX_BOUNDARY.split(text)

    # parts[0] = 文档前言, parts[1] = "ARTICLE 1", parts[2] = 内容, parts[3] = "ARTICLE 2", ...
    if len(parts) > 1:
        raw_blocks.append({"content": parts[0].strip(), "article": "preamble", "title": "Preamble"})
        i = 1
        while i < len(parts) - 1:
            label = parts[i].strip()
            body = parts[i + 1].strip()
            article_num = re.search(r'(\d+)', label)
            article_id = f"Article {article_num.group(1)}" if article_num else label
            raw_blocks.append({"content": body, "article": article_id, "title": label})
            i += 2
    else:
        raw_blocks.append({"content": text, "article": "full", "title": "Full Document"})

    # 在每个大文章块内按章节号切分
    chunks = []
    for block in raw_blocks:
        sub_chunks = split_by_sections(block)
        chunks.extend(sub_chunks)

    return chunks


def split_by_sections(block: dict) -> list[dict]:
    """在文章块内部按小节号切分"""
    content = block["content"]
    # 找到所有小节边界
    matches = list(SECTION_BOUNDARY.finditer(content))
    if not matches:
        return [{"content": content, "article": block["article"],
                 "section": block["article"], "title": block["title"]}]

    chunks = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section_num = match.group(1)
        section_text = content[start:end].strip()

        if len(section_text) < 50:  # 太短则跳过（目录行等）
            continue

        chunks.append({
            "content": section_text,
            "article": block["article"],
            "section": section_num,
            "title": block["title"],
        })

    return chunks


def merge_short_chunks(chunks: list[dict], min_chars: int = 300) -> list[dict]:
    """合并过短的 chunk 到下一个"""
    merged = []
    buffer = None
    for ch in chunks:
        if len(ch["content"]) < min_chars:
            if buffer is None:
                buffer = ch
            else:
                buffer["content"] += "\n" + ch["content"]
                buffer["section"] = buffer["section"].split("/")[0] + "/" + ch["section"]
        else:
            if buffer:
                buffer["content"] += "\n" + ch["content"]
                merged.append(buffer)
                buffer = None
            else:
                merged.append(ch)
    if buffer:
        merged.append(buffer)
    return merged


def split_long_chunks(chunks: list[dict], max_chars: int = CHUNK_MAX_CHARS,
                      overlap: int = CHUNK_OVERLAP_CHARS) -> list[dict]:
    """将过长 chunk 按段落 + 重叠窗口切分"""
    result = []
    for ch in chunks:
        text = ch["content"]
        if len(text) <= max_chars:
            result.append(ch)
            continue
        # 按段落切分
        paragraphs = re.split(r'\n\s*\n', text)
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += "\n\n" + para if current else para
            else:
                if current.strip():
                    result.append({**ch, "content": current.strip()})
                # 新块开头：取上一块的 overlap 字符作为上下文
                if len(current) > overlap:
                    current = current[-overlap:] + "\n\n" + para
                else:
                    current = para
        if current.strip():
            result.append({**ch, "content": current.strip()})
    return result


def parse_and_save():
    """主入口：提取、清洗、分块、保存"""
    print("[1/4] Extracting text from PDF...")
    raw_text = extract_text(PDF_PATH)
    print(f"      Extracted {len(raw_text):,} characters")

    print("[2/4] Cleaning text...")
    clean = clean_text(raw_text)
    print(f"      Cleaned: {len(clean):,} characters")

    print("[3/4] Splitting into chunks...")
    blocks = split_into_blocks(clean)
    print(f"      {len(blocks)} initial blocks")

    blocks = merge_short_chunks(blocks)
    print(f"      {len(blocks)} after merging short chunks")

    chunks = split_long_chunks(blocks)
    print(f"      {len(chunks)} final chunks")

    print("[4/4] Saving parsed chunks...")
    PARSED_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARSED_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 打印统计
    total_chars = sum(len(c["content"]) for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    print(f"\n=== Summary ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average chunk size: {avg_chars:.0f} chars")
    print(f"Saved to: {PARSED_TEXT_PATH}")


if __name__ == "__main__":
    parse_and_save()
