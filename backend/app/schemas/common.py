"""
Common schemas for request/response validation.
"""
from typing import Optional, Any, Dict
from pydantic import BaseModel


class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    service: str
    version: str


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Schema for success responses."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    dataset_id: str
    filename: str
    file_size_bytes: int
    status: str
    message: str