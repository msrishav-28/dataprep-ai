"""
Transformation schemas for request/response validation.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class TransformationBase(BaseModel):
    """Base transformation schema with common fields."""
    operation_type: str = Field(..., max_length=100)
    parameters: Dict[str, Any]


class TransformationCreate(TransformationBase):
    """Schema for transformation creation."""
    dataset_id: UUID
    sequence_order: int


class TransformationPreview(TransformationBase):
    """Schema for transformation preview request."""
    dataset_id: UUID
    affected_columns: Optional[List[str]] = None


class TransformationPreviewResponse(BaseModel):
    """Schema for transformation preview response."""
    preview_data: Dict[str, Any]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    affected_rows: int
    explanation: str


class TransformationInDB(TransformationBase):
    """Schema for transformation data in database."""
    transformation_id: UUID
    dataset_id: UUID
    applied_at: datetime
    sequence_order: int

    class Config:
        from_attributes = True


class Transformation(TransformationInDB):
    """Schema for transformation response."""
    pass


class TransformationPipeline(BaseModel):
    """Schema for transformation pipeline response."""
    dataset_id: UUID
    transformations: List[Transformation]
    total_transformations: int