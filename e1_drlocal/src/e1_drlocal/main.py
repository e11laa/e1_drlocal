#!/usr/bin/env python
"""E1DRLocalFlow: CrewAI Flows による Deep Research ワークフロー

元の LangGraph StateGraph ベースの制御フローを
@start / @listen / @router デコレータで再現する。
"""

import argparse
import asyncio
import concurrent.futures
import datetime
import logging
import os
import re
import sys
import time

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from .state import ResearchState
from .crew import DeepResearchCrew
from .constants import (
    MAX_LOOPS,
    MAX_SOURCES_PER_QUERY,
    VERBOSE_STREAMING,
    DEFAULT_SCOUT_MODEL,
    DEFAULT_COMMANDER_MODEL,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WRITER_MODEL,
    ONLINE_SCOUT_MODEL,
    ONLINE_COMMANDER_MODEL,
    ONLINE_WORKER_MODEL,
    ONLINE_WRITER_MODEL,
    REVIEW_DATA_LIMIT_LOCAL,
    REVIEW_DATA_LIMIT_ONLINE,
    RESEARCH_DIMENSIONS,
    QUERY_FORMAT_INSTRUCTION,
    REPORT_QUALITY_REQUIREMENTS,
    LIGHT_REPORT_QUALITY_REQUIREMENTS,
)
from .utils import (
    extract_urls,
    parse_chapters,
    parse_suggested_queries,
    format_report_references,
)


class DeepResearchFlow(Flow[ResearchState]):
    """Deep Research メインフロー

    状態遷移:
        receive_topic → run_planner → run_researcher → run_reviewer
            → [router] check_quality
                → "sufficient"          → run_outliner → run_writer → validate_sources → save_report
                → "need_more_research"  → run_planner (ループ)
    """

    def __init__(
        self,
        scout_model: str = DEFAULT_SCOUT_MODEL,
        commander_model: str = DEFAULT_COMMANDER_MODEL,
        worker_model: str = DEFAULT_WORKER_MODEL,
        writer_model: str = DEFAULT_WRITER_MODEL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crew_instance = DeepResearchCrew(
            scout_model=scout_model,
            commander_model=commander_model,
            worker_model=worker_model,
            writer_model=writer_model,
        )
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
        self._setup_logging()

    def _setup_logging(self):
        """タイムスタンプ付きのログファイルをセットアップ"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"Log_{self.timestamp}.log")

        # 固定の名前を使用することで、ツール側からも getLogger("e1_drlocal") で参照可能にする
        self.logger = logging.getLogger("e1_drlocal")
        self.logger.setLevel(logging.INFO)

        # ハンドラ設定(重複防止のためクリア)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # ファイル出力用
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(file_handler)

        # コンソール出力用(標準出力への重複を避けるため、既存のprintと棲み分けを検討)
        # 今回はprintをそのまま生かしつつ、logger.info() でファイルにも記録する設計に。
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

    def log(self, message, level="info"):
        """コンソール表示とファイル記録を同時に行うラッパー"""
        if level == "info":
            # コンソールへの重複出力を防ぐため、StreamHandlerがある場合はloggerに任せる
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

    # ==========================================
    # 1. トピック受取(エントリポイント)
    # ==========================================
    @start()
    def receive_topic(self):
        """ユーザからトピックを受け取り、State に設定する"""
        self.log(f"\n🚀 Deep Research (CrewAI Flows) 起動 - {self.current_time}")

        # kickoff(inputs={"topic": "..."}) から受け取る場合
        if self.state.topic:
            topic = self.state.topic
        else:
            topic = input("\n🎯 リサーチトピックを入力してください: ").strip()
            if not topic:
                print("トピックが入力されませんでした。終了します。")
                sys.exit(0)
            self.state.topic = topic

        print(f"\n📋 トピック: {topic}")
        print("=" * 60)
        return topic

    # ==========================================
    # 2. Planner(クエリ生成)
    # ==========================================
    @listen("need_more_research")
    def rerun_planner(self):
        """Reviewer が差し戻した場合の Planner 再実行"""
        return self._execute_planner()

    @listen(receive_topic)
    def run_planner(self):
        """初回の Planner 実行"""
        return self._execute_planner()

    def _execute_planner(self):
        """Planner の共通実行ロジック"""
        start_t = time.time()

        self.state.loop_count += 1
        loop = self.state.loop_count
        self.log(f"\n🔍 [Planner] リサーチプランを策定中... (Loop {loop})")

        # Reviewer 推奨クエリがあればバイパス
        if self.state.suggested_queries and loop > 1:
            self.log(f"   ⚡ Reviewer推奨クエリを直接採用 ({len(self.state.suggested_queries)}件)")
            self.state.queries = self.state.suggested_queries[:6]
            self.state.suggested_queries = []
            return self.state.queries

        # フィードバックセクション構築
        feedback_section = ""
        if self.state.reviewer_feedback:
            feedback_section = f"【司令官からの前回フィードバック】\n{self.state.reviewer_feedback}\n\n"
            if self.state.missing_dimensions:
                feedback_section += (
                    "【特に不足が指摘された調査軸】\n"
                    + "\n".join(f"- {d}" for d in self.state.missing_dimensions)
                    + "\n\n上記の不足軸を優先的にカバーする新しいキーワードクエリを生成せよ。前回と同じクエリを繰り返してはならない。\n"
                )

        # Planner Crew を kickoff (リトライ付き)
        @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
        def _kickoff_planner():
            return self.crew_instance.planning_crew().kickoff(
                inputs={
                    "topic": self.state.topic,
                    "current_time": self.current_time,
                    "feedback_section": feedback_section,
                }
            )

        try:
            result = _kickoff_planner()
            
            # もし Pydantic 出力 (Advanced モード) が得られた場合、正規表現パースをスキップ
            if hasattr(result, "pydantic") and result.pydantic:
                self.state.research_plan = result.pydantic.research_plan
                queries = result.pydantic.queries
                if not queries:
                    self.log("   ⚠️ Pydanticでクエリが空だったためトピックをフォールバックします。", level="warning")
                    queries = [self.state.topic]
                self.state.queries = queries[:6]
                self.log(f"\n   📋 [構造化抽出] 生成クエリ ({len(self.state.queries)}件): {self.state.queries}")
                
                elapsed = time.time() - start_t
                self.state.execution_times["Planner"] = self.state.execution_times.get("Planner", 0) + elapsed
                return self.state.queries
                
            raw_output = result.raw if hasattr(result, "raw") else str(result)
        except Exception as e:
            print(f"   ❌ Planner 実行エラー (リトライ上限到達): {e}")
            raw_output = ""

        # リサーチプランの抽出
        if "RESEARCH_PLAN:" in raw_output:
            plan = raw_output.split("RESEARCH_PLAN:")[1].split("QUERIES:")[0].strip()
            self.state.research_plan = plan

        # クエリの抽出とクリーニング
        query_part = raw_output.split("QUERIES:")[-1] if "QUERIES:" in raw_output else ""
        raw_queries = [q.strip() for q in re.split(r"[,，\n]", query_part) if q.strip()]
        queries = []
        for q in raw_queries:
            cleaned = re.sub(r"^[\d\.\-\s\*]+", "", q).strip(" []'\"`「」")
            if cleaned and 3 < len(cleaned) < 80:
                queries.append(cleaned)
        
        if not queries:
            self.log("   ⚠️ クエリの抽出に失敗しました。トピック自体をフォールバッククエリとします。", level="warning")
            queries = [self.state.topic]
            
        self.state.queries = queries[:6]

        self.log(f"\n   📋 生成クエリ ({len(self.state.queries)}件): {self.state.queries}")
        
        elapsed = time.time() - start_t
        self.state.execution_times["Planner"] = self.state.execution_times.get("Planner", 0) + elapsed
        return self.state.queries

    # ==========================================
    # 3. Researcher(情報収集)
    # ==========================================
    @listen(run_planner)
    def run_researcher_initial(self):
        """初回 Planner 後の Researcher 実行"""
        return self._execute_researcher()

    @listen(rerun_planner)
    def run_researcher_loop(self):
        """ループ時 Planner 後の Researcher 実行"""
        return self._execute_researcher()

    def _execute_researcher(self):
        """Researcher の共通実行ロジック: 並列エージェント実行"""
        start_t = time.time()
        self.log(f"\n🌐 [Researcher] マルチソース並列コンテキスト抽出...")

        all_results = []
        
        def _fetch_for_query(idx, query):
            self.log(f"\n   🔎 クエリ {idx + 1}/{len(self.state.queries)}: {query}")
            
            @retry(wait=wait_exponential(multiplier=1, min=3, max=15), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_researcher():
                return self.crew_instance.research_crew().kickoff(
                    inputs={
                        "research_plan": self.state.research_plan,
                        "query": query,
                    }
                )

            try:
                # 毎回独立した Crew インスタンスを生成して実行
                result = _kickoff_researcher()
                raw_output = result.raw if hasattr(result, "raw") else str(result)
                
                # urls
                found_urls = extract_urls(raw_output)
                return {
                    "idx": idx,
                    "output": f"\n### 解析報告: {query}\n{raw_output}\n",
                    "urls": found_urls
                }
            except Exception as e:
                print(f"   ❌ エラー: {query} → {e}")
                return {
                    "idx": idx,
                    "output": "", # エラーによる無駄なノイズテキストを渡さない
                    "urls": []
                }

        # ThreadPoolExecutor でクエリ数分を並列実行
        if not self.state.queries:
            print("   ⚠️ 有効なクエリがないため Researcher フェーズをスキップします。")
            return self.state.research_data

        is_online = os.environ.get("DEEP_RESEARCH_ONLINE") == "1"
        worker_limit = max(1, min(3, len(self.state.queries))) if is_online else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = [executor.submit(_fetch_for_query, i, query) for i, query in enumerate(self.state.queries)]
            
            # 順番通りにするためソート用の辞書を作成
            results_dict = {}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results_dict[res["idx"]] = res["output"]
                for url in res["urls"]:
                    if url not in self.state.fetched_urls:
                        self.state.fetched_urls.append(url)
            
            # クエリの順番通りに、かつ空ではない結果だけを格納
            for i in range(len(self.state.queries)):
                if i in results_dict and results_dict[i].strip():
                    all_results.append(results_dict[i])

        new_data = "".join(all_results)
        
        # アドバンスドモード指定があり、かつ初回ループではない(既存データがある)場合は圧縮実行
        if os.environ.get("DEEP_RESEARCH_ADVANCED") == "1" and self.state.research_data.strip():
            print("\n🧹 [Synthesizer] 既存データと新規データの重複排除・圧縮を実行します...")
            @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_synthesizer():
                return self.crew_instance.synthesizer_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "existing_data": self.state.research_data,
                        "new_data": new_data,
                    }
                )
            try:
                res = _kickoff_synthesizer()
                self.state.research_data = res.raw if hasattr(res, "raw") else str(res)
                print("   ✨ 圧縮完了")
            except Exception as e:
                print(f"   ❌ Synthesizer エラー: {e} -> 生データをそのまま結合します")
                self.state.research_data += new_data
        else:
            self.state.research_data += new_data

        print(f"\n📊 [Researcher] 合計フェッチ済みURL: {len(self.state.fetched_urls)}件")
        
        elapsed = time.time() - start_t
        self.state.execution_times["Researcher"] = self.state.execution_times.get("Researcher", 0) + elapsed
        return self.state.research_data

    # ==========================================
    # 4. Reviewer(品質評価)
    # ==========================================
    @listen(run_researcher_initial)
    def run_reviewer_initial(self):
        """初回 Researcher 後の Reviewer"""
        return self._execute_reviewer()

    @listen(run_researcher_loop)
    def run_reviewer_loop(self):
        """ループ時 Researcher 後の Reviewer"""
        return self._execute_reviewer()

    def _execute_reviewer(self):
        """Reviewer の共通実行ロジック"""
        start_t = time.time()
        self.log(f"\n⚖️ [Reviewer] 厳格な品質検証... (Loop {self.state.loop_count})")

        limit = REVIEW_DATA_LIMIT_ONLINE if os.environ.get("DEEP_RESEARCH_ONLINE") == "1" else REVIEW_DATA_LIMIT_LOCAL
        data_for_review = self.state.research_data[:limit]

        @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
        def _kickoff_reviewer():
            return self.crew_instance.review_crew().kickoff(
                inputs={
                    "topic": self.state.topic,
                    "current_time": self.current_time,
                    "research_plan": self.state.research_plan,
                    "research_data": data_for_review,
                }
            )

        try:
            result = _kickoff_reviewer()
            critic_text = result.raw if hasattr(result, "raw") else str(result)
        except Exception as e:
            self.log(f"   ❌ Reviewer 実行エラー (リトライ上限到達): {e}", level="error")
            critic_text = "FAIL ❌"

        # PASS/FAIL 判定
        is_pass = "PASS" in critic_text.upper()
        is_fail = "FAIL" in critic_text.upper()
        self.state.is_sufficient = is_pass and not is_fail

        # 不足調査軸の抽出
        missing_dims = []
        if "MISSING_DIMENSIONS:" in critic_text:
            missing_section = critic_text.split("MISSING_DIMENSIONS:")[1]
            for delimiter in ["FEEDBACK:", "SUGGESTED_QUERIES:", "VERDICT:"]:
                if delimiter in missing_section:
                    missing_section = missing_section.split(delimiter)[0]
                    break
            missing_dims = [
                line.strip()
                for line in missing_section.strip().split("\n")
                if line.strip()
            ]
        self.state.missing_dimensions = missing_dims

        # 推奨クエリのパース
        suggested = parse_suggested_queries(critic_text)
        self.state.suggested_queries = suggested
        if suggested and VERBOSE_STREAMING:
            self.log(f"\n   📋 Reviewer推奨クエリ: {suggested}")

        self.state.reviewer_feedback = critic_text

        verdict = "PASS ✅" if self.state.is_sufficient else "FAIL ❌"
        self.log(f"\n   判定: {verdict} (Loop {self.state.loop_count}/{MAX_LOOPS})")

        elapsed = time.time() - start_t
        self.state.execution_times["Reviewer"] = self.state.execution_times.get("Reviewer", 0) + elapsed
        return self.state.is_sufficient

    # ==========================================
    # 5. Router(品質により分岐)
    # ==========================================
    @router(run_reviewer_initial)
    def check_quality_initial(self):
        """初回 Reviewer 後のルーティング"""
        return self._route_decision()

    @router(run_reviewer_loop)
    def check_quality_loop(self):
        """ループ時 Reviewer 後のルーティング"""
        return self._route_decision()

    def _route_decision(self) -> str:
        """ルーティング判定ロジック"""
        if self.state.is_sufficient or self.state.loop_count >= MAX_LOOPS:
            if self.state.loop_count >= MAX_LOOPS and not self.state.is_sufficient:
                self.log(f"\n   ⚠️ ループ上限 ({MAX_LOOPS}) に到達。Outliner へ進みます。", level="warning")
            return "sufficient"
        else:
            self.log(f"\n   🔄 追加調査が必要。Planner へ差し戻します。")
            return "need_more_research"

    # ==========================================
    # 6. Outliner(構成案作成)
    # ==========================================
    @listen("sufficient")
    def run_outliner(self):
        """品質 OK → レポート構成を設計"""
        start_t = time.time()
        self.log("\n🏗️ [Outliner] レポート構成を設計中...")

        limit = REVIEW_DATA_LIMIT_ONLINE if os.environ.get("DEEP_RESEARCH_ONLINE") == "1" else REVIEW_DATA_LIMIT_LOCAL
        data_for_outline = self.state.research_data[:limit]

        @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
        def _kickoff_outliner():
            return self.crew_instance.outline_crew().kickoff(
                inputs={
                    "topic": self.state.topic,
                    "research_data": data_for_outline,
                }
            )

        try:
            result = _kickoff_outliner()
            raw_output = result.raw if hasattr(result, "raw") else str(result)
        except Exception as e:
            self.log(f"   ❌ Outliner 実行エラー (リトライ上限到達): {e}", level="error")
            raw_output = ""
        self.state.outline = raw_output

        chapter_count = raw_output.count("CHAPTER:")
        self.log(f"\n   📑 章構成: CHAPTER: が {chapter_count}回出現")
        
        elapsed = time.time() - start_t
        self.state.execution_times["Outliner"] = self.state.execution_times.get("Outliner", 0) + elapsed
        return raw_output

    # ==========================================
    # 7. Writer(レポート執筆)
    # ==========================================
    @listen(run_outliner)
    def run_writer(self):
        """構成案に基づきレポートを執筆"""
        start_t = time.time()
        self.log("\n📝 [Writer] 最終レポート執筆開始...")

        limit = REVIEW_DATA_LIMIT_ONLINE if os.environ.get("DEEP_RESEARCH_ONLINE") == "1" else REVIEW_DATA_LIMIT_LOCAL
        outline = self.state.outline
        research_data = self.state.research_data[:limit]
        chapters = parse_chapters(outline)
        
        # 章ごとに執筆
        self.log(f"   📑 {len(chapters)}章を段階的に並列執筆します")
        chapter_drafts = [""] * len(chapters) # 結果格納用リスト

        def _filter_relevant_data(chapter_title, full_data):
            """章のタイトルに関連する研究データのみを抽出する(軽量モデルの混乱防止)"""
            if not full_data:
                return ""
            sections = full_data.split("### 解析報告:")
            relevant_sections = []
            
            # 章タイトルのキーワード抽出 (2文字以上の名詞的なもの)
            keywords = [k for k in re.findall(r"[\w]{2,}", chapter_title) if len(k) > 1]
            
            for section in sections:
                if not section.strip():
                    continue
                # キーワードが1つでも含まれていれば採用、または分量が少なければ全部送る
                if any(k.lower() in section.lower() for k in keywords) or len(sections) <= 3:
                    relevant_sections.append("### 解析報告:" + section)
            
            return "\n".join(relevant_sections) if relevant_sections else full_data

        def _write_chapter(idx, chapter):
            relevant_data = _filter_relevant_data(chapter['title'], research_data)
            self.log(f"\n  📖 章 {idx + 1}/{len(chapters)}: {chapter['title'][:50]}... (執筆開始)")
            write_instruction = (
                f"レポート全体の構成：\n{outline}\n\n"
                f"あなたは今、以下の章を執筆してください：\n"
                f"タイトル: {chapter['title']}\n"
                f"内容のヒント: {chapter['description']}\n\n"
                f"【提供された関連データ】\n{relevant_data}\n\n"
                f"注意点：\n"
                f"- 前後の章と重複しないよう、この章の守備範囲に集中すること\n"
                f"- 提供データにある事実(数値・固有名詞)のみを使用すること\n"
                f"- **【引用形式の厳守】** 実際のURLを用いて [1] または [Source: https://...] 形式で明記せよ。提供されたデータに含まれる収集済みのURLをそのまま使用すること。"
            )

            @retry(wait=wait_exponential(multiplier=1, min=3, max=15), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_writer_chapter():
                return self.crew_instance.writing_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "write_instruction": write_instruction,
                        "quality_requirements": quality_reqs,
                        "fetched_urls_list": fetched_urls_str,
                    }
                )

            try:
                result = _kickoff_writer_chapter()
                self.log(f"  🏁 章 {idx + 1} 執筆完了")
                return idx, (result.raw if hasattr(result, "raw") else str(result))
            except Exception as e:
                self.log(f"   ❌ 章 {idx + 1} の執筆エラー (リトライ上限到達): {e}", level="error")
                return idx, f"## {chapter['title']}\n\n(執筆エラー: {e})"

        is_light = os.environ.get("DEEP_RESEARCH_LIGHT") == "1"
        quality_reqs_raw = LIGHT_REPORT_QUALITY_REQUIREMENTS if is_light else REPORT_QUALITY_REQUIREMENTS
        fetched_urls_str = "\n".join([f"  * {url}" for url in self.state.fetched_urls]) if self.state.fetched_urls else "  * (有効なソースなし)"
        
        # 明示的に変数を展開する(CrewAIのテンプレート処理漏れ対策)
        quality_reqs = quality_reqs_raw.format(fetched_urls_list=fetched_urls_str)

        if not chapters:
            # フォールバック: 一括生成
            self.log("   ⚠️ 章パース失敗 → 全体一括生成モード", level="warning")
            write_instruction = (
                f"構成案：\n{outline}\n\n"
                f"収集データ：\n{research_data}\n\n"
                "上記の構成案と収集データに基づき、以下の構造を持つ包括的な分析レポートを執筆せよ：\n"
                "1. # タイトル\n"
                "2. ## 序論(背景・目的・射程)\n"
                "3. ## 本論の各章(4〜6章、各章 ## 見出し付き)\n"
                "4. ## 結論(全体統合の洞察)\n"
                "5. ## 参考文献\n\n"
                "最低5000文字以上のレポートを執筆すること。"
            )
            @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_writer_fallback():
                return self.crew_instance.writing_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "write_instruction": write_instruction,
                        "quality_requirements": quality_reqs,
                        "fetched_urls_list": fetched_urls_str,
                    }
                )

            try:
                result = _kickoff_writer_fallback()
                raw_output = result.raw if hasattr(result, "raw") else str(result)
            except Exception as e:
                self.log(f"   ❌ Writer 一括生成エラー: {e}", level="error")
                raw_output = f"# エラー\nレポートの執筆中にエラーが発生しました: {e}"
            self.state.final_report = raw_output
            self.state.chapter_drafts = [raw_output]
            
            elapsed = time.time() - start_t
            self.state.execution_times["Writer"] = self.state.execution_times.get("Writer", 0) + elapsed
            return raw_output

        # 章ごとに執筆
        self.log(f"   📑 {len(chapters)}章を段階的に並列執筆します")
        chapter_drafts = [""] * len(chapters) # 結果格納用リスト

        def _write_chapter(idx, chapter):
            self.log(f"\n  📖 章 {idx + 1}/{len(chapters)}: {chapter['title'][:50]}... (執筆開始)")
            write_instruction = (
                f"レポート全体の構成：\n{outline}\n\n"
                f"あなたは今、以下の章を執筆してください：\n"
                f"--- 現在の章 ---\n{chapter['raw']}\n---\n\n"
                f"以下の収集データを活用してください：\n{research_data}\n\n"
                "この章のみを執筆せよ。他の章については書かないこと。\n"
                "Markdown形式で、## 見出し から始めること。\n"
                "最低500文字以上で執筆すること。"
            )

            @retry(wait=wait_exponential(multiplier=1, min=3, max=15), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_writer_chapter():
                return self.crew_instance.writing_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "write_instruction": write_instruction,
                        "quality_requirements": quality_reqs,
                        "fetched_urls_list": fetched_urls_str,
                    }
                )

            try:
                result = _kickoff_writer_chapter()
                self.log(f"  🏁 章 {idx + 1} 執筆完了")
                return idx, (result.raw if hasattr(result, "raw") else str(result))
            except Exception as e:
                self.log(f"   ❌ 章 {idx + 1} の執筆エラー (リトライ上限到達): {e}", level="error")
                return idx, f"## {chapter['title']}\n\n(執筆エラー: {e})"

        # 複数章を並列で実行
        is_online = os.environ.get("DEEP_RESEARCH_ONLINE") == "1"
        worker_limit = 2 if is_online else 1  # ローカルは1に固定
        
        self.log(f"   ⚙️ {len(chapters)}章を最大 {worker_limit} 並列で段階的に執筆します (レートリミット回避)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = [executor.submit(_write_chapter, i, chap) for i, chap in enumerate(chapters)]
            for future in concurrent.futures.as_completed(futures):
                idx, text = future.result()
                chapter_drafts[idx] = text

        self.state.chapter_drafts = chapter_drafts

        # 統合パス
        if len(chapter_drafts) <= 1:
            self.log("\n   ℹ️ 単一章のため統合パスをスキップ")
            self.state.final_report = chapter_drafts[0] if chapter_drafts else ""
        else:
            self.log(f"\n  🔗 統合パス: {len(chapter_drafts)}章を1つのレポートに統合...")
            combined = "\n\n".join(chapter_drafts)
            integration_instruction = (
                "以下は各章のドラフトです。これらを統合して1つの完成されたレポートにしてください。\n\n"
                "統合時の作業：\n"
                "1. レポートタイトル(# 見出し)を冒頭に付ける\n"
                "2. 序論が無ければ追加する(テーマの背景、本レポートの目的、分析の射程を提示)\n"
                "3. 結論が無ければ追加する(全体の知見を統合した高次の洞察)\n"
                "4. 章間の論理的な接続を確保する(遷移文で自然につなげる)\n"
                "5. 引用文献リストを末尾にまとめる\n"
                "6. 各章のドラフトの内容は保持し、削除しないこと\n\n"
                f"--- 各章のドラフト ---\n{combined}\n---\n\n"
                "上記を最終レポートとして統合せよ。"
            )

            @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_writer_integrate():
                return self.crew_instance.writing_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "write_instruction": integration_instruction,
                        "quality_requirements": quality_reqs,
                        "fetched_urls_list": fetched_urls_str,
                    }
                )

            try:
                result = _kickoff_writer_integrate()
                self.state.final_report = result.raw if hasattr(result, "raw") else str(result)
            except Exception as e:
                print(f"   ❌ 統合パスエラー: {e}")
                self.state.final_report = f"# エラー (統合パス)\n\n{combined}"

        # ====== [Phase 4] Editorial Review (推敲ループ) ======
        if os.environ.get("DEEP_RESEARCH_ADVANCED") == "1":
            print("\n🧐 [Editor] 最終レポート案の品質監査を実行します...")
            
            @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
            def _kickoff_editor(draft):
                return self.crew_instance.editor_crew().kickoff(
                    inputs={
                        "topic": self.state.topic,
                        "draft_report": draft,
                    }
                )
                
            draft = self.state.final_report
            try:
                editor_res = _kickoff_editor(draft)
                editor_text = editor_res.raw if hasattr(editor_res, "raw") else str(editor_res)
                
                if "VERDICT: FAIL" in editor_text.upper():
                    print("   ⚠️ Editor からの指摘 (FAIL): 修正リライトを実行します")
                    feedback = editor_text.split("FAIL", 1)[-1].strip()
                    revise_instruction = (
                        f"以下のレポート草案には編集長からダメ出し(修正指示)が入りました。\n"
                        f"指摘に厳密に従い、該当部分を修正して完全なレポートを再出力せよ。\n\n"
                        f"【編集長からの指摘】\n{feedback}\n\n"
                        f"【現在のドラフト】\n{draft}"
                    )
                    
                    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
                    def _kickoff_rewrite():
                        return self.crew_instance.writing_crew().kickoff(
                            inputs={
                                "topic": self.state.topic,
                                "write_instruction": revise_instruction,
                            }
                        )
                    
                    rewrite_res = _kickoff_rewrite()
                    self.state.final_report = rewrite_res.raw if hasattr(rewrite_res, "raw") else str(rewrite_res)
                    print("   ✅ リライト完了")
                else:
                    print("   ✅ Editor 検査 PASS: 問題ありませんでした")
            except Exception as e:
                print(f"   ❌ Editor 実行エラー: {e} -> 元のドラフトを採用します")

        elapsed = time.time() - start_t
        self.state.execution_times["Writer"] = self.state.execution_times.get("Writer", 0) + elapsed
        return self.state.final_report

    # ==========================================
    # 8. ソース検証
    # ==========================================
    @listen(run_writer)
    def validate_sources(self):
        """レポート内の引用URLを検証し、引用番号スタイルに変換 (utilsへ委譲)"""
        self.log("\n🔎 [SourceValidator] ソース信頼性の検証と引用スタイル変換中...")
        is_strict = os.environ.get("DEEP_RESEARCH_STRICT_SOURCES") == "1"
        self.state.final_report = format_report_references(
            self.state.final_report, 
            self.state.fetched_urls,
            strict_sources=is_strict
        )
        return self.state.final_report

    # ==========================================
    # 9. レポート保存
    # ==========================================
    @listen(validate_sources)
    def save_report(self):
        """最終レポートをファイルに保存し、実行サマリーを追記"""
        output_dir = "research_reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"Report_{self.timestamp}_crewai.md"
        filepath = os.path.join(output_dir, filename)

        # 実行サマリーの構築 (Markdown)
        total_time = sum(self.state.execution_times.values())
        summary = f"\n\n---\n## 実行サマリー (Execution Summary)\n"
        summary += f"- **実行日時**: {self.current_time}\n"
        
        # 主要な引数の記録
        if self.state.cli_args:
            args = self.state.cli_args
            summary += f"- **実行オプション**: "
            options = []
            for opt in ["light", "online", "advanced", "strict_sources"]:
                if args.get(opt):
                    options.append(f"`--{opt}`")
            summary += (", ".join(options) if options else "デフォルト") + "\n"
            summary += f"- **使用モデル**:\n"
            summary += f"  - Scout: `{args.get('scout') or DEFAULT_SCOUT_MODEL}`\n"
            summary += f"  - Commander: `{args.get('commander') or DEFAULT_COMMANDER_MODEL}`\n"
            summary += f"  - Worker: `{args.get('worker') or DEFAULT_WORKER_MODEL}`\n"
            summary += f"  - Writer: `{args.get('writer') or args.get('commander') or DEFAULT_WRITER_MODEL}`\n"

        summary += f"- **所要時間総計**: {total_time:.1f}秒 ({total_time/60:.1f}分)\n"
        summary += f"  - **フェッチ済みURL数**: {len(self.state.fetched_urls)}件\n"
        summary += f"  - **リサーチループ回数**: {self.state.loop_count}回\n"
        summary += f"  - **セクション別所要時間**:\n"
        for task, duration in self.state.execution_times.items():
            summary += f"    - {task}: {duration:.2f}秒\n"

        # レポート本体に追記
        self.state.final_report += summary

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.state.final_report)

        self.log(f"\n✨ レポート完成: {filepath}")
        self.log("=" * 60)
        
        # コンソール/ログ用の統計表示
        self.log("\n📊 実行統計:")
        self.log(f"   - 文字数: {len(self.state.final_report)}字")
        for task, duration in self.state.execution_times.items():
            self.log(f"   - {task}: {duration:.2f}秒")
        self.log(f"   - 合計: {total_time:.1f}秒")
        
        return filepath


# ==========================================
# CLI エントリポイント
# ==========================================
def kickoff():
    """crewai flow kickoff / uv run kickoff から呼ばれるエントリポイント"""
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Deep Research (CrewAI Flows)")

    parser.add_argument(
        "--scout", type=str, default=DEFAULT_SCOUT_MODEL,
        help="先遣隊 (Planner/Outliner) のモデルを指定",
    )
    parser.add_argument(
        "--commander", type=str, default=DEFAULT_COMMANDER_MODEL,
        help="司令塔 (Reviewer/Writer) のモデルを指定",
    )
    parser.add_argument(
        "--worker", type=str, default=DEFAULT_WORKER_MODEL,
        help="作業員 (Researcher) のモデルを指定",
    )
    parser.add_argument(
        "--writer", type=str, default=DEFAULT_WRITER_MODEL,
        help="執筆者 (Writer) のモデルを指定(省略時は司令塔と同じ)",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="軽量ローカルモデル構成で実行する",
    )
    parser.add_argument(
        "--online", action="store_true",
        help="クラウド (OpenRouter) モデルで実行する(OPENROUTER_API_KEY が必要)",
    )
    parser.add_argument(
        "--advanced", action="store_true",
        help="高度な機能(構造化出力・情報圧縮・推敲ループ)を強制有効にする(--online時は自動有効)",
    )
    parser.add_argument(
        "--strict-sources", action="store_true",
        help="ハルシネーション対策: 検索成功したURL以外からの引用を物理削除する",
    )
    parser.add_argument(
        "--topic", type=str, default="",
        help="リサーチトピック(省略時は対話式入力)",
    )

    args = parser.parse_args()

    # フラグの環境変数へのバインド
    if args.light:
        os.environ["DEEP_RESEARCH_LIGHT"] = "1"
        # 軽量モデルは自動で strict モードを有効にする(CLIで明示的にオフにする機能はない前提)
        args.strict_sources = True

    if args.strict_sources:
        os.environ["DEEP_RESEARCH_STRICT_SOURCES"] = "1"

    if args.advanced or args.online:
        os.environ["DEEP_RESEARCH_ADVANCED"] = "1"
        print("🪄 アドバンスドモード作動: (構造化出力 / 情報圧縮 / 推敲ループが有効になります)")

    if args.online:
        # オンラインフラグを環境変数にセット(WebFetchToolなどから参照できるようにするため)
        os.environ["DEEP_RESEARCH_ONLINE"] = "1"
        
        # uv run が .env を自動で環境変数に展開している前提で、存在確認のみを行うかチェック
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("❌ OPENROUTER_API_KEY が設定されていません。")
            print("   .env ファイルに設定してください。")
            sys.exit(1)

        print("🌐 OpenRouter経由でクラウドモデルを使用します。")
        args.scout = ONLINE_SCOUT_MODEL
        args.commander = ONLINE_COMMANDER_MODEL
        args.worker = ONLINE_WORKER_MODEL
        args.writer = ONLINE_WRITER_MODEL
    elif args.light:
        print("⚡ 軽量プロファイル(--light)が指定されました。")
        args.scout = "ollama/gemma3n:e2b"
        args.commander = "ollama/gemma3n:e2b"
        args.worker = "ollama/qwen2.5:3b"
        args.writer = "ollama/qwen2.5:3b"

    if args.strict_sources:
        print("🛡️ 【Strict Sources】検証済み以外の架空URL引用を強制削除します。")

    print("-" * 60)
    print(f"🤖 [Scout]     : {args.scout}")
    print(f"🤖 [Commander] : {args.commander}")
    print(f"🤖 [Worker]    : {args.worker}")
    print(f"🤖 [Writer]    : {args.writer}")
    print("-" * 60)

    flow = DeepResearchFlow(
        scout_model=args.scout,
        commander_model=args.commander,
        worker_model=args.worker,
        writer_model=args.writer,
    )
    # 引数をシリアライズ可能な形で状態に保存
    flow.state.cli_args = vars(args)

    # トピックが指定されていればinputsに渡す
    inputs = {}
    if args.topic:
        inputs["topic"] = args.topic

    flow.kickoff(inputs=inputs)


def plot():
    """フローの可視化"""
    flow = DeepResearchFlow()
    flow.plot("e1_drlocal_flow")


if __name__ == "__main__":
    kickoff()
