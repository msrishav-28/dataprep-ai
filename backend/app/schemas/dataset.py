"""
Dataset schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class DatasetBase(BaseModel):
    """Base dataset schema with common fields."""
    filename: str = Field(..., max_length=255)


class DatasetCreate(DatasetBase):
    """Schema for dataset creation."""
    file_path: str = Field(..., max_length=500)
    file_size_bytes: Optional[int] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    column_metadata: Optional[Dict[str, Any]] = None


class DatasetUpdate(BaseModel):
    """Schema for dataset updates."""
    filename: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, max_length=50)
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    column_metadata: Optional[Dict[str, Any]] = None


class DatasetInDB(DatasetBase):
    """Schema for dataset data in database."""
    dataset_id: UUID
    user_id: UUID
    file_path: str
    file_size_bytes: Optional[int] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    upload_date: datetime
    status: str
    column_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class Dataset(DatasetInDB):
    """Schema for dataset response."""
    pass


class DatasetPreview(BaseModel):
    """Schema for dataset preview response."""
    dataset_id: UUID
    filename: str
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    sample_data: Optional[Dict[str, Any]] = None
    column_info: Optional[Dict[str, Any]] = None