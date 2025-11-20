import sys

from loguru import logger

from app.core.config import settings
from app.domain.enums import BaseEnum


class LogLevel(BaseEnum):
    TRACE = "TRACE"  # Detailed debug information. Enable only during development, disable in production.
    DEBUG = "DEBUG"  # Debug information. Enable during troubleshooting, disable in production.
    INFO = "INFO"  # Records normal system operations. Always enabled.
    WARNING = (
        "WARNING"  # Events requiring attention. Monitor in operations, alert if 3/min.
    )
    ERROR = "ERROR"  # Processing failures or exceptions. Requires immediate investigation. Alert if 1/min.
    CRITICAL = "CRITICAL"  # System outages or critical failures. Requires immediate response and recovery. Alert if 1/min.


def set_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, level=settings.LOG_LEVEL)


def log(log_level: LogLevel, subject: str, object: str, message: str) -> None:
    logger.log(log_level.value, f"[{subject}] {object} | {message}")
