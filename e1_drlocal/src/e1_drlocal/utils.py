"""ユーティリティ関数: 元スクリプトのヘルパーを移行"""

import re
from typing import List

from .constants import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """テキストをオーバーラップ付きでチャンク分割する"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_urls(raw_result) -> list[str]:
    """SearxNG の検索結果から URL を堅牢に抽出する（複数形式に対応）"""
    urls: list[str] = []

    # パターン1: dict のリスト（SearxNG JSON 形式）
    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict):
                for key in ["url", "href", "link"]:
                    if key in item and item[key]:
                        urls.append(str(item[key]))
                        break

    # パターン2: 単一の dict に results キー
    if isinstance(raw_result, dict):
        results = raw_result.get("results", raw_result.get("data", []))
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    for key in ["url", "href", "link"]:
                        if key in item and item[key]:
                            urls.append(str(item[key]))
                            break

    # パターン3: 文字列から URL 抽出（汎用フォールバック）
    text = str(raw_result)
    found = re.findall(r'https?://[^\s\'"<>}\]\),]+', text)
    for u in found:
        u = u.rstrip(".;:")
        if u not in urls and len(u) > 15:
            urls.append(u)

    # 重複除去（順序保持）
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    return unique


def parse_chapters(outline: str) -> list[dict]:
    """複数フォーマットの章構成を堅牢にパースする"""
    chapters: list[dict] = []

    # パターン1: CHAPTER: 形式
    if "CHAPTER:" in outline:
        parts = outline.split("CHAPTER:")
        for part in parts[1:]:
            title = part.split("\n")[0].strip()
            if title:
                chapters.append({"title": title, "raw": f"CHAPTER: {part}"})

    # パターン2: Markdown ## 見出し形式（フォールバック）
    if len(chapters) < 2:
        chapters = []
        lines = outline.split("\n")
        current_chapter = None
        current_content: list[str] = []
        for line in lines:
            match = re.match(r"^#{1,2}\s+(?:\d+[\.\\)]\s*)?(.+)", line)
            if match:
                if current_chapter:
                    chapters.append(
                        {"title": current_chapter, "raw": "\n".join(current_content)}
                    )
                current_chapter = match.group(1).strip()
                current_content = [line]
            elif current_chapter:
                current_content.append(line)
        if current_chapter:
            chapters.append(
                {"title": current_chapter, "raw": "\n".join(current_content)}
            )

    return chapters if len(chapters) >= 2 else []


def parse_suggested_queries(critic_text: str) -> list[str]:
    """Reviewer の出力から SUGGESTED_QUERIES をパースする"""
    suggested: list[str] = []
    if "SUGGESTED_QUERIES:" not in critic_text:
        return suggested

    sq_section = critic_text.split("SUGGESTED_QUERIES:")[1]

    backtick_queries = re.findall(r"`([^`]+)`", sq_section)
    bracket_queries = re.findall(r"「([^」]+)」", sq_section)
    quoted_queries = re.findall(r'"([^"]+)"', sq_section)

    suggested = backtick_queries + bracket_queries + quoted_queries

    if not suggested:
        for line in sq_section.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
            if (
                cleaned
                and len(cleaned) > 5
                and not cleaned.startswith("**")
                and not cleaned.startswith("*(")
            ):
                suggested.append(cleaned)

    suggested = [q for q in suggested if len(q) < 80]
    return suggested[:8]


def format_report_references(report: str, fetched_urls: list[str], strict_sources: bool = False) -> str:
    """レポート内の引用を整理し、番号付き引用 [1] に統一して末尾にリストを追加する。"""
    valid_urls_list = fetched_urls if fetched_urls else []
    valid_urls_set = set(valid_urls_list)

    # 1. 本文から既存の [Source: URL] 形式を抽出
    cited_urls = re.findall(r"\[Source:\s*(https?://[^\]\s]+)\]", report)
    
    # 2. LLMが直接書いた可能性のある末尾のURLリストや、本文中の裸のURLも可能な限り拾う
    # (ただし、参考文献一覧セクションなどのURLを優先的に拾う)
    plain_urls = re.findall(r"(https?://[^\s)\]\"'>]+)", report)
    
    # 重複を排除しつつ、出現順（本文中の引用優先）を保持
    unique_cited = []
    for url in (cited_urls + plain_urls):
        url = url.strip(".,")
        if url not in unique_cited:
            # 信頼性を高めるため、基本的には fetched_urls にあるものを優先
            unique_cited.append(url)

    # 3. strict_sources が True の場合、未調査URLへの参照を削除または警告付きにする
    if strict_sources:
        unique_cited = [u for u in unique_cited if u in valid_urls_set]
        # 本文中の無効な [Source: URL] を消去
        all_found = re.findall(r"\[Source:\s*(https?://[^\]\s]+)\]", report)
        for u in all_found:
            if u not in valid_urls_set:
                report = report.replace(f"[Source: {u}]", "")

    # 4. マッピング作成
    url_mapping = {url: i for i, url in enumerate(unique_cited, 1)}

    # 5. 本文の置換: [Source: URL] -> [index]
    for url, idx in url_mapping.items():
        # [Source: URL] 形式を置換
        report = re.sub(r"\[Source:\s*" + re.escape(url) + r"\s*\]", f"[{idx}]", report)
        # もしLLMがすでに [1] [2] と書いていて、かつURLが裸で書かれている場合、
        # 裸のURLを消去しつつ、番号が整合するようにしたいが、誤爆リスクがあるため
        # ここでは最低限 [Source: URL] の置換にとどめる。
    
    # 6. 既存の「参考文献」セクションがあれば、一旦削除して再構築する
    report = re.split(r"\n\n---\n## 参考文献一覧", report)[0]
    report = re.split(r"\n## 参考文献", report)[0]
    report = report.strip()

    # 7. 参考文献リストの構築
    if unique_cited:
        references = "\n\n---\n## 参考文献一覧\n\n"
        for url, idx in url_mapping.items():
            valid_mark = "" if url in valid_urls_set else " （※LLMによる外部知識引用）"
            references += f"{idx}. {url}{valid_mark}\n"
        report += references

    return report
