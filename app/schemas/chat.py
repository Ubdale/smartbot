from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    # Start a new conversation about this dataset...
    dataset_id: str | None = None
    # ...or continue an existing one (follow-ups reuse its dataset and previous plan).
    conversation_id: str | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question cannot be blank")
        return v


class DateRangeOut(BaseModel):
    start: str
    end: str  # inclusive last day, for display
    label: str


class ChartOut(BaseModel):
    type: Literal["bar", "line", "pie"]
    x: str
    y: str
    series: str | None = None


class TableOut(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class ExplanationOut(BaseModel):
    """'How this was calculated': everything needed to audit the numbers."""

    description: str
    sql: str
    params: list[Any]
    date_range: DateRangeOut | None
    plan: dict[str, Any]
    cached: bool
    elapsed_ms: int
    # A caveat about the data itself, e.g. the table was read from an image by AI.
    data_note: str | None = None


class AnswerOut(BaseModel):
    """A computed answer over a table."""

    type: Literal["table"] = "table"
    table: TableOut
    chart: ChartOut | None
    explanation: ExplanationOut
    summary_source: Literal["llm", "template"]
    reused: bool = False  # answered instantly from an identical earlier question


class SourceOut(BaseModel):
    n: int  # the number the answer cites, e.g. [1]
    page: int | None
    text: str
    cited: bool


class DocumentAnswerOut(BaseModel):
    """An answer quoted from a document, with the passages it came from."""

    type: Literal["document"]
    grounded: bool  # False: the model's answer couldn't be backed up and was withheld
    sources: list[SourceOut]


class CodeBlockOut(BaseModel):
    language: str
    code: str


class CodeAnswerOut(BaseModel):
    """Code that reproduces an earlier answer (generated from its plan, not by the model)."""

    type: Literal["code"]
    blocks: list[CodeBlockOut]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["answered", "clarification", "failed"] | None
    answer: CodeAnswerOut | DocumentAnswerOut | AnswerOut | None
    rating: int | None = None  # 1 helpful, -1 wrong
    created_at: datetime


class AskResponse(BaseModel):
    conversation_id: str
    question: MessageOut
    reply: MessageOut


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    dataset_id: str
    created_at: datetime
    updated_at: datetime


class ConversationOut(ConversationSummary):
    messages: list[MessageOut]


class FeedbackRequest(BaseModel):
    rating: Literal[-1, 0, 1]  # 0 clears it


class FeedbackOut(BaseModel):
    message_id: str
    rating: int | None
    learned: bool  # saved as an example future questions can learn from
