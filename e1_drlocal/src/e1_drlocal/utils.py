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


def format_report_references(report: str, fetched_urls: list[str]) -> str:
    """レポート内の [Source: URL] を番号付き引用 [1] に置換し、末尾にリストを追加する"""
    valid_urls = set(fetched_urls)

    # レポート内の全引用URLを抽出（順序保持）
    cited_urls_all = re.findall(r"\[Source:\s*(https?://[^\]\s]+)\]", report)
    
    # 重複を削除しつつ出現順を保持
    unique_cited = []
    for url in cited_urls_all:
        if url not in unique_cited:
            unique_cited.append(url)

    url_mapping = {}
    for i, url in enumerate(unique_cited, 1):
        url_mapping[url] = i

    # 本文中の置換
    for url, idx in url_mapping.items():
        report = report.replace(f"[Source: {url}]", f"[{idx}]")
        report = re.sub(r"\[Source:\s*" + re.escape(url) + r"\s*\]", f"[{idx}]", report)

    # 参考文献リスト
    if unique_cited:
        references = "\n\n---\n## 参考文献一覧\n\n"
        for url in unique_cited:
            idx = url_mapping[url]
            valid_mark = "" if url in valid_urls else " （※未検証ソース・LLMのハルシネーションの可能性）"
            references += f"{idx}. {url}{valid_mark}\n"
        
        if "### フェッチ検証済みソース一覧" in report:
            report = report.split("### フェッチ検証済みソース一覧")[0].strip()
        
        report += references

    return report
