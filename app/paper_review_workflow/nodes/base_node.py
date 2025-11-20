"""Base class for paper review workflow nodes."""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from app.paper_review_workflow.models.state import PaperReviewAgentState


class BaseNode(ABC):
    """Base class for workflow nodes.
    
    All nodes must inherit from this class and implement the __call__ method.
    """
    
    def __init__(self) -> None:
        """Initialize base node."""
        self.logger = logger
    
    @abstractmethod
    def __call__(self, state: PaperReviewAgentState) -> dict[str, Any]:
        """Execute node processing.
        
        Args:
        ----
            state: Current workflow state
            
        Returns:
        -------
            Dictionary of state updates
        """
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Get node name."""
        return self.__class__.__name__
    
    def log_start(self, message: str) -> None:
        """Output node start log."""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_success(self, message: str) -> None:
        """Output node success log."""
        self.logger.success(f"[{self.name}] {message}")
    
    def log_error(self, message: str) -> None:
        """Output node error log."""
        self.logger.error(f"[{self.name}] {message}")
    
    def log_warning(self, message: str) -> None:
        """Output node warning log."""
        self.logger.warning(f"[{self.name}] {message}")
    
    def log_debug(self, message: str) -> None:
        """Output node debug log."""
        self.logger.debug(f"[{self.name}] {message}")

