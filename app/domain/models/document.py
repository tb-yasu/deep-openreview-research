from pydantic import BaseModel, Field

from app.domain.enums import ManagedTaskStatus
from app.core.utils.datetime_utils import get_current_time
from app.core.utils.nano_id import NanoID, generate_id


class Document(BaseModel):
    id: NanoID = Field(title="Document ID", default_factory=generate_id)
    title: str = Field(title="Title")
    url: str = Field(title="URL")
    abstract: str = Field(title="Abstract", default="")
    authors: list[str] = Field(title="Authors", default_factory=list)

    def to_string(self) -> str:
        return f"""\
<document>
    <id>{self.id}</id>
    <title>{self.title}</title>
    <link>{self.url}</link>
    <abstract>{self.abstract}</abstract>
    <authors>{", ".join(self.authors)}</authors>
</document>"""


class ManagedDocument(Document):
    task_id: NanoID = Field(title="Task ID")
    status: ManagedTaskStatus = Field(
        title="Document Status", default=ManagedTaskStatus.NOT_STARTED
    )
    summary: str | None = Field(
        title="Detailed Summary for Task Resolution",
        description=(
            "Describe in specific and detailed terms the important knowledge, insights, or solutions "
            "related to the task this document is responsible for. "
            "The summary should clearly include the document's main claims, supporting data or theories, "
            "and points that directly contribute to task completion. "
            "Also specify any differentiation from other documents or unique contributions to the task."
        ),
        default=None,
    )
    created_at: str = Field(default_factory=get_current_time)
    updated_at: str = Field(default_factory=get_current_time)

    def to_string(self) -> str:
        return f"""\
<document>
    <id>{self.id}</id>
    <title>{self.title}</title>
    <link>{self.url}</link>
    <abstract>{self.abstract}</abstract>
    <authors>{", ".join(self.authors)}</authors>
    <summary>{self.summary}</summary>
</document>"""
