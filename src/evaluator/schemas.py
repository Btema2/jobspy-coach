import math
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


class JobEvaluationResult(BaseModel):
    score: float = Field(ge=1.0, le=10.0, description="Match score between 1.0 and 10.0 (out-of-range values are clamped)")
    is_wow: bool = Field(default=False, description="Set by the pipeline from the final score; the LLM need not provide it")
    summary: str = Field(
        min_length=10,
        validation_alias=AliasChoices("summary", "summary_uk"),
        description="1-2 sentence executive summary (language set by the profile); the old key summary_uk is accepted",
    )
    reasoning: str = Field(description="English reasoning")
    language_requirements: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    hard_reject: bool = Field(default=False, description="Role matches a hard-reject rule of the rubric")
    matches_bonus_criteria: bool = Field(default=False, description="Posting explicitly values the bonus criteria described in the local profile")
    is_core_role: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_core_role", "role_is_direct_coaching"),
        description="Role is a core role of the profile; the old key role_is_direct_coaching is accepted",
    )
    eval_model: Optional[str] = Field(default=None, description="Model that actually produced the evaluation")
    raw_score: Optional[float] = Field(
        default=None,
        description="Clamped model score before bonus, reject cap, title safety net and non-core cap (set by the pipeline)",
    )

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> Any:
        """Clamp numeric scores into [1.0, 10.0]; leave non-numeric input for normal validation to reject."""
        if isinstance(value, bool):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if math.isnan(number):
            return value
        return min(10.0, max(1.0, number))
