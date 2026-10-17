from typing import List, Optional
from pydantic import BaseModel, Field


class JobEvaluationResult(BaseModel):
    score: float = Field(ge=1.0, le=10.0, description="Match score between 1.0 and 10.0")
    is_wow: bool = Field(description="Flag for exceptional dream roles")
    summary_uk: str = Field(min_length=10, description="1-2 sentence Ukrainian summary")
    reasoning: str = Field(description="English reasoning")
    language_requirements: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    eval_model: Optional[str] = Field(default=None, description="Actual model routed by OpenRouter")

