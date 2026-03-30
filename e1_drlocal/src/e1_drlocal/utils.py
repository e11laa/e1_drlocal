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
    """SearxNG の検索結果から URL を堅牢に抽出する(複数形式に対応)"""
    urls: list[str] = []

    # パターン1: dict のリスト(SearxNG JSON 形式)
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

    # パターン3: 文字列から URL 抽出(汎用フォールバック)
    text = str(raw_result)
    found = re.findall(r'https?://[^\s\'"<>}\]\),]+', text)
    for u in found:
        u = u.rstrip(".;:")
        if u not in urls and len(u) > 15:
            urls.append(u)

    # 重複除去(順序保持)
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

    # パターン2: Markdown ## 見出し形式(フォールバック)
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


def format_report_references(report: str, fetched_urls: list[str], strict_sources: bool = False, is_light: bool = False) -> str:
    """レポート内の引用を整理し、番号付き引用 [1] に統一して末尾にリストを追加する。"""
    
    # 0. 【ライトモード限定】サニタイジング（変数の漏洩防止）
    if is_light:
        # {fetched_urls_list} や {quality_requirements} などのテンプレート変数を強制削除
        report = re.sub(r'\{[a-zA-Z0-9_]+\}', '', report)
        # AIが勝手に作った壊れたMarkdownアンカー [#0] などを [1] 形式に修復
        report = re.sub(r'\[(\d+)\]\(#\d+\)', r'[\1]', report)

    valid_urls_list = fetched_urls if fetched_urls else []
    valid_urls_set = set(valid_urls_list)

    # 1. 本文から引用タグを抽出
    # 標準: [Source: URL], ライトモード特有: [[REF: URL]]
    cited_standard = re.findall(r"\[Source:\s*(https?://[^\s\]]+)\]", report)
    cited_ref_tag = re.findall(r"\[\[REF:\s*(https?://[^\s\]]+)\]\]", report)
    
    # 2. LLMが直接書いた可能性のあるURLも拾う
    plain_urls = re.findall(r"(https?://[^\s)\]\"'>]+)", report)
    
    # 重複を排除しつつ、出現順（本文中の引用優先）を保持
    unique_cited = []
    for url in (cited_standard + cited_ref_tag + plain_urls):
        url = url.strip("., ")
        if url and url not in unique_cited:
            unique_cited.append(url)

    # 3. strict_sources が True の場合、未調査URLへの参照を削除
    if strict_sources:
        unique_cited = [u for u in unique_cited if u in valid_urls_set]
        # 本文中の無効なタグを消去
        report = re.sub(r"\[Source:\s*(https?://[^\s\]]+)\]", 
                        lambda m: m.group(0) if m.group(1) in valid_urls_set else "", report)
        report = re.sub(r"\[\[REF:\s*(https?://[^\s\]]+)\]\]", 
                        lambda m: m.group(0) if m.group(1) in valid_urls_set else "", report)

    # 4. マッピング作成
    url_mapping = {url: i for i, url in enumerate(unique_cited, 1)}

    # 5. 本文の置換: タグ -> [index]
    for url, idx in url_mapping.items():
        # 標準タグ置換
        report = re.sub(r"\[Source:\s*" + re.escape(url) + r"\s*\]", f"[{idx}]", report)
        # ライトモード用タグ置換
        report = re.sub(r"\[\[REF:\s*" + re.escape(url) + r"\s*\]\]", f"[{idx}]", report)
    
    # 6. 既存の「参考文献」セクションがあれば、一旦削除して再構築する
    #    末尾から検索し、見出しレベルの揺れ (##, ###) や改行数の差異に対応。
    #    本文中に偶然「参考文献」が出現しても、最後のものだけを切り取ることで本文消失を防ぐ。
    ref_match = None
    for m in re.finditer(r"\n+(?:---\n)?#{2,3}\s*参考文献", report):
        ref_match = m  # 最後のマッチを保持
    if ref_match:
        report = report[:ref_match.start()]
    report = report.strip()

    # 7. 参考文献リストの構築
    if unique_cited:
        references = "\n\n---\n## 参考文献一覧\n\n"
        for url, idx in url_mapping.items():
            valid_mark = "" if url in valid_urls_set else " (※LLMによる外部知識引用)"
            references += f"{idx}. {url}{valid_mark}\n"
        report += references

    return report
