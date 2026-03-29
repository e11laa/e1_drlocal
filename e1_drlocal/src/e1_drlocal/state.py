"""ResearchState: Flowの構造化状態管理モデル"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    """Deep Research Flow 全体で共有される状態。
    元の LangGraph TypedDict を Pydantic BaseModel に移行。
    """

    topic: str = ""
    research_plan: str = ""
    queries: List[str] = Field(default_factory=list)
    research_data: str = ""
    fetched_urls: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    outline: str = ""
    chapter_drafts: List[str] = Field(default_factory=list)
    loop_count: int = 0
    is_sufficient: bool = False
    final_report: str = ""
    reviewer_feedback: str = ""
    missing_dimensions: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    execution_times: dict = Field(default_factory=dict)
    cli_args: dict = Field(default_factory=dict)
