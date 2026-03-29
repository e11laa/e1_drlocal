"""SearxNG 検索カスタムツール"""

import json
import logging
import os
import re
from typing import Type

import requests
import litellm
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..constants import (
    SEARXNG_URL,
    RE_RANK_MAX_RESULTS,
    RE_RANK_TOP_K,
    RERANKER_MODEL_ONLINE,
    RERANKER_MODEL_LOCAL,
)


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

    def _rerank_results(self, query: str, results: list[dict]) -> list[dict]:
        """LLM-as-a-Judge による検索結果の再ランク付け"""
        logger = logging.getLogger("e1_drlocal")
        if not results:
            return []

        logger.info(f"   ⚖️ [Reranker] {len(results)}件の結果から最適なソースを選別中...")
        
        is_online = os.environ.get("DEEP_RESEARCH_ONLINE") == "1"
        model = RERANKER_MODEL_ONLINE if is_online else RERANKER_MODEL_LOCAL
        
        # セレクション用のコンテキスト作成
        candidates_text = ""
        for i, res in enumerate(results):
            candidates_text += f"ID: {i}\nTitle: {res['title']}\nSnippet: {res['content'][:200]}\nURL: {res['url']}\n---\n"

        prompt = f"""あなたは優秀なリサーチ選別官です。
ユーザーのリサーチクエリに対して、提供された検索結果リストの中から、最も情報の信頼性が高く、具体的で、調査の核心に迫るURLを最大{RE_RANK_TOP_K}件だけ選んでください。

リサーチクエリ: {query}

【候補リスト】
{candidates_text}

【指示】
- クエリの意図に最も合致するURLを選んでください。
- 広告サイト、無関係なトップページ、中身のないまとめサイトは除外してください。
- 選択したURLのみを、以下のJSON形式で返してください。
- 出力はJSON以外を含めないでください。

例:
["https://example.com/page1", "https://example.com/page2"]
"""

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"} if "ollama" not in model else None,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            
            # JSON 抽出 (qwen などの小型モデルが余計な文言を付けた場合の対策)
            match = re.search(r"(\[.*\])", content, re.DOTALL)
            if match:
                selected_urls = json.loads(match.group(1))
            else:
                selected_urls = json.loads(content)

            # 選択されたURLに一致する結果を抽出
            reranked = [res for res in results if res["url"] in selected_urls]
            
            # LLMが1件も選ばなかった、またはURLを捏造した場合のフォールバック
            if not reranked:
                logger.warning("   ⚠️ [Reranker] 有効なURLが選別されませんでした。上位のみを採用します。")
                return results[:RE_RANK_TOP_K]

            logger.info(f"   ✅ [Reranker] 選別完了: {len(reranked)}件をフェッチ対象に決定")
            return reranked[:RE_RANK_TOP_K]

        except Exception as e:
            logger.error(f"   ❌ [Reranker] エラーによりスキップ: {e}")
            return results[:10]  # エラー時は上位10件をそのまま返す

    def _run(self, query: str) -> str:
        """SearxNG API を呼び出して検索結果を返す"""
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "language": "auto",
                "pageno": 1,
            }
            resp = requests.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            raw_results = []
            for item in data.get("results", []):
                raw_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            # リランキングの実行
            if len(raw_results) > RE_RANK_TOP_K:
                # 最大 RE_RANK_MAX_RESULTS 件を対象にリランク
                to_rerank = raw_results[:RE_RANK_MAX_RESULTS]
                results = self._rerank_results(query, to_rerank)
            else:
                results = raw_results

            return json.dumps(results, ensure_ascii=False, indent=2)

        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"SearxNG 検索エラー: {str(e)}"}, ensure_ascii=False)
