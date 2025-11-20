"""Paper Review Agent implementation."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.core.logging import LogLevel
from app.domain.base_agent import LangGraphAgent
from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    PaperReviewAgentInputState,
    PaperReviewAgentOutputState,
)
from app.paper_review_workflow.nodes import (
    GatherResearchInterestsNode,
    SearchPapersNode,
    EvaluatePapersNode,  # Keep for initial filtering
    RankPapersNode,
    ReRankPapersNode,
    GeneratePaperReportNode,
)
from app.paper_review_workflow.nodes.unified_llm_evaluate_papers_node import (
    UnifiedLLMEvaluatePapersNode,
)
from app.paper_review_workflow.config import LLMConfig, ScoringWeights, DEFAULT_SCORING_WEIGHTS


class PaperReviewAgent(LangGraphAgent):
    """Paper Review Agent.
    
    Uses the OpenReview API to search and analyze accepted papers from conferences,
    extracting valuable papers based on user's research interests.
    """
    
    def __init__(
        self,
        checkpointer: MemorySaver | None = None,
        log_level: LogLevel = LogLevel.INFO,
        recursion_limit: int = 100,
        llm_config: LLMConfig | None = None,
        scoring_weights: ScoringWeights | None = None,
        top_n: int | None = None,
    ) -> None:
        """Initialize PaperReviewAgent.
        
        Args:
        ----
            checkpointer: Checkpointer for state persistence
            log_level: Logging level
            recursion_limit: Maximum recursion limit
            llm_config: LLM evaluation configuration (uses default if omitted)
            scoring_weights: Scoring weight configuration (uses default if omitted)
            top_n: Number of papers to include in final report (same as top_k if omitted)
        """
        weights = scoring_weights or DEFAULT_SCORING_WEIGHTS
        
        self.gather_interests_node = GatherResearchInterestsNode()
        self.search_papers_node = SearchPapersNode()
        self.evaluate_papers_node = EvaluatePapersNode(scoring_weights=weights)  # Initial filtering
        self.rank_papers_node = RankPapersNode(llm_config=llm_config)
        # Unified LLM evaluation (calculates all scores in one call)
        self.unified_llm_evaluate_node = UnifiedLLMEvaluatePapersNode(
            llm_config=llm_config,
            scoring_weights=weights,
        )
        # Always pass top_n (default value is used if None)
        self.re_rank_papers_node = ReRankPapersNode(top_n=top_n if top_n is not None else 9999)
        self.generate_report_node = GeneratePaperReportNode()
        
        super().__init__(
            log_level=log_level,
            checkpointer=checkpointer,
            recursion_limit=recursion_limit,
        )
    
    def _create_graph(self) -> CompiledStateGraph:
        """Create workflow graph.
        
        Returns:
        -------
            Compiled StateGraph
        """
        workflow = StateGraph(
            state_schema=PaperReviewAgentState,
            input_schema=PaperReviewAgentInputState,
            # Remove output_schema to return full state (format output here in the future)
        )
        
        # Add nodes
        workflow.add_node("gather_interests", self.gather_interests_node)
        workflow.add_node("search_papers", self.search_papers_node)
        workflow.add_node("evaluate_papers", self.evaluate_papers_node)  # Initial filtering
        workflow.add_node("rank_papers", self.rank_papers_node)
        # Unified LLM evaluation (calculates all scores in one call)
        workflow.add_node("unified_llm_evaluate", self.unified_llm_evaluate_node)
        workflow.add_node("re_rank_papers", self.re_rank_papers_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # Define workflow edges
        workflow.add_edge("gather_interests", "search_papers")
        workflow.add_edge("search_papers", "evaluate_papers")
        workflow.add_edge("evaluate_papers", "rank_papers")
        workflow.add_edge("rank_papers", "unified_llm_evaluate")  # Unified LLM evaluation
        workflow.add_edge("unified_llm_evaluate", "re_rank_papers")
        workflow.add_edge("re_rank_papers", "generate_report")
        
        # Set entry and finish points
        workflow.set_entry_point("gather_interests")
        workflow.set_finish_point("generate_report")
        
        return workflow.compile(checkpointer=self.checkpointer)


def create_graph(
    llm_config: LLMConfig | None = None,
    scoring_weights: ScoringWeights | None = None,
    top_n: int | None = None,
) -> CompiledStateGraph:
    """Create PaperReviewAgent graph.
    
    Args:
    ----
        llm_config: LLM evaluation configuration (uses default if omitted)
        scoring_weights: Scoring weight configuration (uses default if omitted)
        top_n: Number of papers to include in final report (same as top_k if omitted)
    
    Returns:
    -------
        Compiled graph
    """
    checkpointer = InMemorySaver()
    agent = PaperReviewAgent(
        checkpointer=checkpointer,
        log_level=LogLevel.DEBUG,
        recursion_limit=100,
        llm_config=llm_config,
        scoring_weights=scoring_weights,
        top_n=top_n,
    )
    return agent.graph


def invoke_graph(
    graph: CompiledStateGraph,
    input_data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute graph.
    
    Args:
    ----
        graph: Graph to execute
        input_data: Input data
        config: Execution configuration
        
    Returns:
    -------
        Execution result
    """
    if config is None:
        config = {"recursion_limit": 100, "thread_id": "default"}
    
    logger.info("Starting PaperReviewAgent execution...")
    result = graph.invoke(
        input=input_data,
        config=config,
    )
    logger.info("PaperReviewAgent execution completed")
    
    return result

