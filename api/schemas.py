from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ClassScore(Camel):
    class_id: str
    crop: str
    condition: str
    display_name: str
    healthy: bool
    confidence: float


class PredictResponse(Camel):
    request_id: str
    model_version: str
    status: Literal["CONFIDENT", "LOW_CONFIDENCE"]
    prediction: Optional[ClassScore] = None      # null when LOW_CONFIDENCE
    candidates: List[ClassScore]                 # top-k, always present
    confidence_threshold: float
    quality_warnings: List[str]
    message: str
    disclaimer: str
    inference_ms: float


class ClassInfo(Camel):
    class_id: str
    crop: str
    condition: str
    display_name: str
    healthy: bool


class ErrorBody(Camel):
    code: str
    message: str


class ErrorResponse(Camel):
    request_id: str
    error: ErrorBody
