from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    """Minimal schema for validating incoming user input."""

    message: str = Field(..., description="The user's prompt or message.")
    selected_tomes: list[str] | None = Field(
        default=None,
        description="Optional list of specific book/file names to filter retrieval by.",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model override to use for this request.",
    )
    think: bool | None = Field(
        default=None,
        description="Optional flag to explicitly enable or disable chain-of-thought thinking.",
    )


class RuleCitation(BaseModel):
    """Citation information for a consulted rulebook passage."""

    source: str = Field(..., description="Rulebook or document name.")
    section: str = Field("General", description="Section or chapter heading.")
    page: int | None = Field(None, description="Page number if available.")
    text: str = Field(..., description="Excerpt of the rule text.")


class AIResponse(BaseModel):
    """Structured response output from the AI service layer."""

    content: str = Field(
        ...,
        description="The generated response text or user-friendly error message.",
    )
    thinking: str | None = Field(
        None,
        description="The model's internal reasoning or chain-of-thought trace, if produced.",
    )
    success: bool = Field(True, description="Flag indicating if the operation succeeded.")
    error_message: str | None = Field(
        None, description="Detailed error description if success is False."
    )
    model: str | None = Field(
        None, description="Model used to provide the answer or None if no model was used."
    )
    citations: list[RuleCitation] = Field(
        default_factory=list,
        description="Rule citations used to formulate the response.",
    )


class IngestionResponse(BaseModel):
    """Schema for document ingestion response."""

    filename: str = Field(..., description="Name of the ingested file.")
    chunks_count: int = Field(
        0, description="Number of rule chunks parsed and indexed."
    )
    success: bool = Field(True, description="Flag indicating if ingestion succeeded.")
    error_message: str | None = Field(
        None, description="Detailed error description if success is False."
    )
