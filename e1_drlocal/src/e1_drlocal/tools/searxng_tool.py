"""SearxNG 検索カスタムツール"""

import json
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..constants import SEARXNG_URL


class SearxNGSearchInput(BaseModel):
    """SearxNG 検索ツールの入力スキーマ"""
    query: str = Field(..., description="検索キーワード（スペース区切り）")


class SearxNGSearchTool(BaseTool):
    name: str = "searxng_web_search"
    description: str = (
        "SearxNG メタ検索エンジンを使って Web 検索を行い、結果を JSON で返す。"
        "キーワードをスペース区切りで入力すること。"
    )
    args_schema: Type[BaseModel] = SearxNGSearchInput
    searxng_url: str = SEARXNG_URL

    def _run(self, query: str) -> str:
        """SearxNG API を呼び出して検索結果を返す"""
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "language": "auto",
            }
            resp = requests.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:10]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            return json.dumps(results, ensure_ascii=False, indent=2)

        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"SearxNG 検索エラー: {str(e)}"}, ensure_ascii=False)
