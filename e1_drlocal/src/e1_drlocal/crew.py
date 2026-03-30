from pathlib import Path
from typing import Optional, List  # List を追加

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
        *,
        is_online: bool = False,
        is_light: bool = False,
        is_advanced: bool = False,
    ):
        self.scout_model = scout_model
        self.commander_model = commander_model
        self.worker_model = worker_model
        self.writer_model = writer_model
        self.is_online = is_online
        self.is_light = is_light
        self.is_advanced = is_advanced

        # YAML 設定のロード
        self._agents_cfg = _load_yaml("agents.yaml")
        self._tasks_cfg = _load_yaml("tasks.yaml")

        # 共有ツールの初期化
        self._search_tool = SearxNGSearchTool(is_online=self.is_online, is_light=self.is_light)
        self._web_fetch_tool = WebFetchTool(is_online=self.is_online)

        # 逐次実行用エージェントの事前生成 (再利用可能)
        self._planner_agent = self._create_planner_agent()
        self._reviewer_agent = self._create_reviewer_agent()
        self._outliner_agent = self._create_outliner_agent()
        self._synthesizer_agent = self._create_synthesizer_agent()
        self._editor_agent = self._create_editor_agent()

    # ==========================================
    # エージェント構築ロジック (内部用)
    # ==========================================

    def _create_planner_agent(self) -> Agent:
        cfg = self._agents_cfg["planner"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def _create_reviewer_agent(self) -> Agent:
        cfg = self._agents_cfg["reviewer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.commander_model,
            verbose=True,
        )

    def _create_outliner_agent(self) -> Agent:
        cfg = self._agents_cfg["outliner"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def _create_synthesizer_agent(self) -> Agent:
        cfg = self._agents_cfg["synthesizer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.scout_model,
            verbose=True,
        )

    def _create_editor_agent(self) -> Agent:
        cfg = self._agents_cfg["editor"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.commander_model,
            verbose=True,
        )

    # ==========================================
    # 公開エージェント取得メソッド (Flowから呼び出し)
    # ==========================================

    def planner(self) -> Agent:
        return self._planner_agent

    def researcher(self) -> Agent:
        """並列実行されるため、呼び出しごとに個別のインスタンスを生成する (スレッドセーフティ確保)"""
        cfg = self._agents_cfg["researcher"]
        
        # --light の場合のみ Temperature を下げる
        llm = self.worker_model
        if self.is_light:
            llm = LLM(model=self.worker_model, temperature=0.1)

        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=llm,
            tools=[self._search_tool, self._web_fetch_tool],
            verbose=True,
            max_iter=5,  # ツール利用の無限ループ防止
            max_retry_limit=2,
        )

    def reviewer(self) -> Agent:
        return self._reviewer_agent

    def outliner(self) -> Agent:
        return self._outliner_agent

    def writer(self) -> Agent:
        """多章並列執筆される場合があるため、都度生成を選択 (安全性優先)"""
        cfg = self._agents_cfg["writer"]
        return Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=self.writer_model,
            verbose=True,
        )

    def synthesizer(self) -> Agent:
        return self._synthesizer_agent

    def editor(self) -> Agent:
        return self._editor_agent

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
        
        if self.is_advanced:
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

