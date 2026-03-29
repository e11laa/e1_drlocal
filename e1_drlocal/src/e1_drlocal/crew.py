import os
from pathlib import Path
from typing import Optional, List  # List を追加

import sys
import yaml
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, Field  # 追加: 構造化出力用

from .tools.searxng_tool import SearxNGSearchTool
from .tools.web_fetch_tool import WebFetchTool
from .constants import (
    DEFAULT_SCOUT_MODEL,
    DEFAULT_COMMANDER_MODEL,
    DEFAULT_WORKER_MODEL,
    DEFAULT_WRITER_MODEL,
)

# === 追加: Planner抽出用のデータスキーマ ===
class PlannerOutput(BaseModel):
    research_plan: str = Field(..., description="現状の分析と今後のリサーチ方針(思考プロセス)")
    queries: List[str] = Field(..., description="次に実行すべき具体的な検索クエリのリスト(最大6つ)")
# ==============================================

# ==============================================

def _load_yaml(filename: str) -> dict:
    """config/ ディレクトリから YAML ファイルを読み込む"""
    config_dir = Path(__file__).parent / "config"
    with open(config_dir / filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


class DeepResearchCrew:
    """Deep Research の Crew 定義。
    agents.yaml と tasks.yaml を手動で読み込み、5 エージェント・5 タスクを構成する。
    @CrewBase は使用せず、通常のクラスとして実装。
    """

    def __init__(
        self,
        scout_model: str = DEFAULT_SCOUT_MODEL,
        commander_model: str = DEFAULT_COMMANDER_MODEL,
        worker_model: str = DEFAULT_WORKER_MODEL,
        writer_model: str = DEFAULT_WRITER_MODEL,
    ):
        self.scout_model = scout_model
        self.commander_model = commander_model
        self.worker_model = worker_model
        self.writer_model = writer_model

        # YAML 設定のロード
        self._agents_cfg = _load_yaml("agents.yaml")
        self._tasks_cfg = _load_yaml("tasks.yaml")

    # ==========================================
    # エージェント定義
    # ==========================================

    def planner(self) -> Agent:
        cfg = self._agents_cfg["planner"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def researcher(self) -> Agent:
        cfg = self._agents_cfg["researcher"]
        
        # --light の場合のみ Temperature を下げる
        llm = self.worker_model
        if "--light" in sys.argv:
            llm = LLM(model=self.worker_model, temperature=0.1)
            
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=llm,
            tools=[SearxNGSearchTool(), WebFetchTool()],
            verbose=True,
            max_iter=5,  # ツール利用の無限ループ防止
            max_retry_limit=2,
        )

    def reviewer(self) -> Agent:
        cfg = self._agents_cfg["reviewer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.commander_model,
            verbose=True,
        )

    def outliner(self) -> Agent:
        cfg = self._agents_cfg["outliner"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def writer(self) -> Agent:
        cfg = self._agents_cfg["writer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.writer_model,
            verbose=True,
        )

    def synthesizer(self) -> Agent:
        cfg = self._agents_cfg["synthesizer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def editor(self) -> Agent:
        cfg = self._agents_cfg["editor"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.commander_model,
            verbose=True,
        )

    # ==========================================
    # タスク定義(エージェントを引数で渡す)
    # ==========================================

    def plan_research_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["plan_research"]
        
        task_args = {
            "description": cfg["description"],
            "expected_output": cfg["expected_output"],
            "agent": agent,
        }
        
        if os.environ.get("DEEP_RESEARCH_ADVANCED") == "1":
            task_args["output_pydantic"] = PlannerOutput
            
        return Task(**task_args)

    def research_topic_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["research_topic"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    def review_research_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["review_research"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    def create_outline_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["create_outline"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    def write_report_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["write_report"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    def synthesize_data_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["synthesize_data"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    def edit_report_task(self, agent: Agent) -> Task:
        cfg = self._tasks_cfg["edit_report"]
        return Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agent,
        )

    # ==========================================
    # Crew 構築メソッド群(Flow から呼び出される)
    # ==========================================

    def planning_crew(self) -> Crew:
        """Planner エージェントだけの Crew"""
        _planner = self.planner()
        return Crew(
            agents=[_planner],
            tasks=[self.plan_research_task(_planner)],
            process=Process.sequential,
            verbose=True,
        )

    def research_crew(self) -> Crew:
        """Researcher エージェントだけの Crew"""
        _researcher = self.researcher()
        return Crew(
            agents=[_researcher],
            tasks=[self.research_topic_task(_researcher)],
            process=Process.sequential,
            verbose=True,
        )

    def review_crew(self) -> Crew:
        """Reviewer エージェントだけの Crew"""
        _reviewer = self.reviewer()
        return Crew(
            agents=[_reviewer],
            tasks=[self.review_research_task(_reviewer)],
            process=Process.sequential,
            verbose=True,
        )

    def outline_crew(self) -> Crew:
        """Outliner エージェントだけの Crew"""
        _outliner = self.outliner()
        return Crew(
            agents=[_outliner],
            tasks=[self.create_outline_task(_outliner)],
            process=Process.sequential,
            verbose=True,
        )

    def writing_crew(self) -> Crew:
        """Writer エージェントだけの Crew"""
        _writer = self.writer()
        return Crew(
            agents=[_writer],
            tasks=[self.write_report_task(_writer)],
            process=Process.sequential,
            verbose=True,
        )

    def synthesizer_crew(self) -> Crew:
        """Synthesizer エージェントだけの Crew"""
        _synthesizer = self.synthesizer()
        return Crew(
            agents=[_synthesizer],
            tasks=[self.synthesize_data_task(_synthesizer)],
            process=Process.sequential,
            verbose=True,
        )

    def editor_crew(self) -> Crew:
        """Editor エージェントだけの Crew"""
        _editor = self.editor()
        return Crew(
            agents=[_editor],
            tasks=[self.edit_report_task(_editor)],
            process=Process.sequential,
            verbose=True,
        )

