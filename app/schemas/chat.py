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


class AnswerOut(BaseModel):
    table: TableOut
    chart: ChartOut | None
    explanation: ExplanationOut
    summary_source: Literal["llm", "template"]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["answered", "clarification", "failed"] | None
    answer: AnswerOut | None
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
