"""Web フェッチカスタムツール"""

from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import os
from ..constants import FETCH_TIMEOUT, FETCH_TEXT_LIMIT_LOCAL, FETCH_TEXT_LIMIT_ONLINE


class WebFetchInput(BaseModel):
    """Web フェッチツールの入力スキーマ"""
    url: str = Field(..., description="フェッチ対象の URL")


class WebFetchTool(BaseTool):
    name: str = "web_fetch"
    description: str = (
        "指定した URL の Web ページを取得し、テキストとして返す。"
        "HTML タグを除去した本文テキストを返す。"
    )
    args_schema: Type[BaseModel] = WebFetchInput
    
    def _truncate_text(self, text: str, limit: int) -> str:
        """指定文字数制限手前の「句点」や「改行」でセマンティックに切り詰める"""
        if len(text) <= limit:
            return text
            
        # 制限文字数から遡って、句点や改行を探す
        # 日本語(。)、英語(.)、改行(\n)
        truncated = text[:limit]
        last_punct = -1
        for punct in ["\n", "。", "."]:
            pos = truncated.rfind(punct)
            if pos > last_punct:
                last_punct = pos
        
        # 句点が見つかった場合、その直後で切る。見つからない場合は物理カットのフォールバック
        if last_punct != -1:
            return text[:last_punct + 1] + "\n\n[... セマンティックに切り詰め ...]"
        else:
            return text[:limit] + "\n\n[... 以降省略 ...]"

    def _run(self, url: str) -> str:
        """URL からコンテンツを取得してテキスト化する"""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ja,en;q=0.5",
            }

            limit = FETCH_TEXT_LIMIT_ONLINE if os.environ.get("DEEP_RESEARCH_ONLINE") == "1" else FETCH_TEXT_LIMIT_LOCAL

            # === 1. Jina Reader API での取得を試行 (SPA/JS対応 & 高品質Markdown) ===
            try:
                jina_url = f"https://r.jina.ai/{url}"
                jina_resp = requests.get(jina_url, headers=headers, timeout=FETCH_TIMEOUT)
                if jina_resp.status_code == 200:
                    text = self._truncate_text(jina_resp.text, limit)
                    return text
                else:
                    # HTTPエラー時はフォールバックへ
                    pass
            except requests.RequestException as e:
                # ネットワーク例外の場合は原因をログ出ししてフォールバックへ進む
                print(f"   [WebFetch] Jina API Timeout/Error ({url}), falling back to direct fetch.")

            # === 2. フォールバック: 標準のHTML取得とBeautifulSoup ===
            resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"

            # BeautifulSoup でテキスト抽出
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")

                # 不要な要素を除去
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                # BeautifulSoup が無い場合は正規表現でフォールバック
                import re
                text = resp.text
                # scriptとstyleタグの中身(JavaScriptやCSSのコード)を先に消去
                text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.IGNORECASE | re.DOTALL)
                # 残りのHTMLタグを消去
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\s+", " ", text).strip()

            # テキストが長すぎる場合は切り詰め
            text = self._truncate_text(text, limit)

            return text

        except requests.exceptions.Timeout:
            return f"エラー: タイムアウト ({FETCH_TIMEOUT}秒) - {url}"
        except requests.exceptions.RequestException as e:
            return f"エラー: フェッチ失敗 - {url} → {str(e)[:200]}"
