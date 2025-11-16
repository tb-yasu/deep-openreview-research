"""Node for ranking evaluated papers."""

import asyncio
import re
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from app.paper_review_workflow.models.state import (
    PaperReviewAgentState,
    EvaluatedPaper,
    EvaluationCriteria,
)
from app.paper_review_workflow.utils import convert_papers_to_dict_list
from app.paper_review_workflow.constants import (
    MAX_DISPLAY_PAPERS,
    PRELIMINARY_LLM_MAX_TOKENS,
    ABSTRACT_SHORT_LENGTH,
    MAX_KEYWORDS_DISPLAY,
)
from app.paper_review_workflow.config import LLMConfig


class RankPapersNode:
    """評価済み論文をスコア順にランク付けするノード."""
    
    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        """RankPapersNodeを初期化.
        
        Args:
        ----
            llm_config: LLM設定（並列数などを含む）
        """
        self.llm = None  # 必要時に初期化（コスト削減）
        self.llm_config = llm_config
    
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """論文ランキングを実行.
        
        Args:
        ----
            state: 現在の状態
            
        Returns:
        -------
            更新された状態の辞書
        """
        logger.info(f"Ranking {len(state.evaluated_papers)} evaluated papers...")
        
        # 評価基準に基づいてフィルタリング
        criteria = state.evaluation_criteria
        filtered_papers = [
            paper for paper in state.evaluated_papers
            if self._meets_criteria(paper, criteria)
        ]
        
        logger.info(
            f"After filtering: {len(filtered_papers)}/{len(state.evaluated_papers)} papers "
            f"meet the criteria"
        )
        
        # 総合スコアでソート（降順）
        ranked_papers = sorted(
            filtered_papers,
            key=lambda p: p.overall_score or 0.0,
            reverse=True,
        )
        
        # 簡易LLMフィルタ（有効な場合）
        if criteria.enable_preliminary_llm_filter and len(ranked_papers) > 0:
            logger.info("🔍 Preliminary LLM filter enabled - evaluating top candidates...")
            ranked_papers = self._apply_preliminary_llm_filter(
                ranked_papers, 
                criteria
            )
        
        # top_kが指定されている場合、上位k件のみ選択
        if criteria.top_k_papers is not None:
            selected_papers = ranked_papers[:criteria.top_k_papers]
            logger.info(
                f"Selected top {criteria.top_k_papers} papers from {len(ranked_papers)} ranked papers "
                f"(actual: {len(selected_papers)})"
            )
        else:
            selected_papers = ranked_papers
            logger.info(f"All {len(ranked_papers)} papers selected (no top_k limit)")
        
        # 上位論文を辞書形式に変換（表示用）
        top_papers = convert_papers_to_dict_list(
            selected_papers,
            max_count=MAX_DISPLAY_PAPERS,
            include_llm_scores=False,
        )
        
        if top_papers:
            logger.success(f"Top paper: {top_papers[0]['title'][:50]} (Score: {top_papers[0]['overall_score']:.3f})")
        
        return {
            "ranked_papers": selected_papers,  # LLM評価に渡す論文リスト
            "top_papers": top_papers,
        }
    
    def _meets_criteria(self, paper: EvaluatedPaper, criteria: EvaluationCriteria) -> bool:
        """論文が評価基準を満たすかチェック.
        
        Args:
        ----
            paper: 評価済み論文
            criteria: 評価基準
            
        Returns:
        -------
            基準を満たす場合True
        """
        # 関連性スコアの最小値チェック
        if paper.relevance_score is not None:
            if paper.relevance_score < criteria.min_relevance_score:
                return False
        
        # レビュースコアの最小値チェック
        if criteria.min_rating is not None and paper.rating_avg is not None:
            if paper.rating_avg < criteria.min_rating:
                return False
        
        return True
    
    def _apply_preliminary_llm_filter(
        self, 
        ranked_papers: list[EvaluatedPaper], 
        criteria: EvaluationCriteria,
    ) -> list[EvaluatedPaper]:
        """簡易LLM評価でrelevance_scoreを再計算し、再ソート（並列処理版）.
        
        Args:
        ----
            ranked_papers: ソート済み論文リスト
            criteria: 評価基準
            
        Returns:
        -------
            relevance_scoreを更新して再ソートした論文リスト
        """
        # 評価対象数を決定
        filter_count = min(
            criteria.preliminary_llm_filter_count,
            len(ranked_papers)
        )
        
        # 並列数を決定（llm_configがあればそれを使用、なければデフォルト10）
        max_concurrent = self.llm_config.max_concurrent if self.llm_config else 10
        
        logger.info(f"Evaluating top {filter_count} papers with LLM for better relevance scoring...")
        logger.info(f"⚡ Parallel execution with max {max_concurrent} concurrent requests")
        
        # LLM初期化
        if self.llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=PRELIMINARY_LLM_MAX_TOKENS,
            )
        
        # 上位N件を並列LLM評価
        target_papers = ranked_papers[:filter_count]
        updated_papers = asyncio.run(
            self._evaluate_relevance_parallel(target_papers, criteria, max_concurrent=max_concurrent)
        )
        
        # 残りの論文（LLM評価しない）を追加
        remaining_papers = ranked_papers[filter_count:]
        all_papers = updated_papers + remaining_papers
        
        # relevance_scoreで再ソート（overall_scoreに反映されているので、overall_scoreでソート）
        re_ranked_papers = sorted(
            all_papers,
            key=lambda p: p.overall_score or 0.0,
            reverse=True,
        )
        
        # 成功数をカウント（デフォルトスコアでない論文）
        success_count = sum(1 for p in updated_papers if p.relevance_score != (ranked_papers[0].relevance_score or 0.0))
        
        logger.success(
            f"✓ Preliminary LLM filter completed: {success_count}/{filter_count} papers re-scored (parallel)"
        )
        
        return re_ranked_papers
    
    async def _evaluate_relevance_parallel(
        self,
        papers: list[EvaluatedPaper],
        criteria: EvaluationCriteria,
        max_concurrent: int = 10,
    ) -> list[EvaluatedPaper]:
        """複数論文の関連性を並列評価.
        
        Args:
        ----
            papers: 評価対象論文リスト
            criteria: 評価基準
            max_concurrent: 最大同時実行数
            
        Returns:
        -------
            更新された論文リスト
        """
        # Semaphoreで同時実行数を制限
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(paper, index, total):
            async with semaphore:
                return await self._evaluate_single_relevance_async(paper, criteria, index, total)
        
        # 全論文を並列実行
        tasks = [
            evaluate_with_semaphore(paper, i + 1, len(papers))
            for i, paper in enumerate(papers)
        ]
        
        # 全タスクを実行（エラーが発生しても他のタスクは継続）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 正常に完了した論文のみを返す（Exceptionは除外）
        updated_papers = [
            result for result in results
            if not isinstance(result, Exception)
        ]
        
        # エラーが発生した論文数をログ
        error_count = len(results) - len(updated_papers)
        if error_count > 0:
            logger.warning(f"⚠ {error_count}/{len(results)} papers failed during relevance evaluation")
        
        return updated_papers
    
    async def _evaluate_single_relevance_async(
        self,
        paper: EvaluatedPaper,
        criteria: EvaluationCriteria,
        index: int,
        total: int,
    ) -> EvaluatedPaper:
        """単一論文の関連性を非同期で評価.
        
        Args:
        ----
            paper: 評価対象論文
            criteria: 評価基準
            index: 論文番号（ログ用）
            total: 総論文数（ログ用）
            
        Returns:
        -------
            更新された論文
        """
        try:
            # LLMで関連性を評価（非同期）
            llm_relevance = await self._evaluate_relevance_with_llm_async(paper, criteria)
            
            # relevance_scoreを更新
            updated_paper = paper.model_copy(deep=True)
            old_score = paper.relevance_score or 0.0
            updated_paper.relevance_score = llm_relevance
            
            # overall_scoreも更新（relevance_weightを考慮）
            score_diff = llm_relevance - old_score
            updated_paper.overall_score = (paper.overall_score or 0.0) + score_diff * 0.4  # relevance_weight=0.4
            
            if index % 50 == 0:
                logger.info(f"  Progress: {index}/{total} papers evaluated")
            
            return updated_paper
            
        except Exception as e:
            logger.warning(f"Failed to LLM evaluate paper {paper.id}: {e}")
            # 失敗時は元のスコアを保持
            return paper
    
    async def _evaluate_relevance_with_llm_async(
        self,
        paper: EvaluatedPaper,
        criteria: EvaluationCriteria,
    ) -> float:
        """LLMで論文の関連性を簡易評価（非同期版）.
        
        Args:
        ----
            paper: 評価対象論文
            criteria: 評価基準
            
        Returns:
        -------
            関連性スコア（0.0-1.0）
        """
        # アブストラクトを短縮
        abstract_short = (
            paper.abstract[:ABSTRACT_SHORT_LENGTH] + 
            ("..." if len(paper.abstract) > ABSTRACT_SHORT_LENGTH else "")
        )
        
        # ユーザーの興味を文字列化
        interests_str = ", ".join(criteria.research_interests)
        user_description = criteria.research_description or f"Keywords: {interests_str}"
        
        # プロンプトを作成
        prompt = f"""
Evaluate how relevant the following paper is to the user's research interests on a scale of 0.0-1.0.

# Paper Information

**Title**: {paper.title}

**Keywords**: {', '.join(paper.keywords[:MAX_KEYWORDS_DISPLAY])}

**Abstract**:
{abstract_short}

# User's Research Interests

{user_description}

# Output Format

Output only the score in the range of 0.0-1.0 (e.g., 0.85)
"""
        
        # LLMに非同期で問い合わせ
        response = await self.llm.ainvoke(prompt)
        response_text = response.content.strip()
        
        # スコアを抽出
        score_match = re.search(r'(0\.\d+|1\.0|0|1)', response_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        else:
            logger.warning(f"Failed to parse relevance score from: {response_text[:100]}")
            return paper.relevance_score or 0.5
    
    def _evaluate_relevance_with_llm(
        self, 
        paper: EvaluatedPaper, 
        criteria: EvaluationCriteria,
    ) -> float:
        """LLMで論文の関連性を簡易評価.
        
        Args:
        ----
            paper: 評価対象論文
            criteria: 評価基準
            
        Returns:
        -------
            関連性スコア（0.0-1.0）
        """
        # アブストラクトを短縮
        abstract_short = (
            paper.abstract[:ABSTRACT_SHORT_LENGTH] + 
            ("..." if len(paper.abstract) > ABSTRACT_SHORT_LENGTH else "")
        )
        keywords_str = ", ".join(paper.keywords[:MAX_KEYWORDS_DISPLAY])
        
        # research_description がない場合は research_interests をフォールバック
        research_interests_str = ", ".join(criteria.research_interests)
        user_interests = criteria.research_description or f"Keywords: {research_interests_str}"
        
        prompt = f"""Rate the relevance of this paper to the user's research interests.

User's Research Interests:
{user_interests}

Paper:
Title: {paper.title}
Keywords: {keywords_str}
Abstract: {abstract_short}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly addresses the research interests
- 0.7-0.9: Very relevant, closely related
- 0.4-0.6: Moderately relevant, some overlap
- 0.1-0.3: Slightly relevant, tangential connection
- 0.0: Not relevant

Return ONLY a single number between 0.0 and 1.0 (e.g., "0.85"). No other text.
"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # 数値を抽出
            # "0.85"のような形式、または"The relevance is 0.85"のような形式に対応
            match = re.search(r'(\d+\.?\d*)', response_text)
            if match:
                score = float(match.group(1))
                # 0-1の範囲に制限
                score = max(0.0, min(1.0, score))
                return score
            else:
                logger.warning(f"Could not parse LLM response: {response_text[:50]}")
                return paper.relevance_score or 0.5
                
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return paper.relevance_score or 0.5

