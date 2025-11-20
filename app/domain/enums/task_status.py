from app.domain.enums.base import BaseEnum


class TaskStatus(BaseEnum):
    COMPLETED = "completed"  # Completed
    FAILED = "failed"  # Failed
    PENDING = "pending"  # Pending


class ManagedTaskStatus(BaseEnum):
    NOT_STARTED = "not_started"  # Not started
    IN_PROGRESS = "in_progress"  # In progress
    COMPLETED = TaskStatus.COMPLETED.value  # Completed
    PENDING = TaskStatus.PENDING.value  # Pending
    FAILED = TaskStatus.FAILED.value  # Failed
