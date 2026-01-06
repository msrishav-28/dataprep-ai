"""
Analysis schemas for request/response validation.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel


class QualityIssue(BaseModel):
    """Schema for data quality issues."""
    issue_type: str
    severity: str
    column: Optional[str] = None
    description: str
    count: int
    percentage: Optional[float] = None


class Recommendation(BaseModel):
    """Schema for data quality recommendations."""
    issue_type: str
    recommendation: str
    explanation: str
    priority: int
    affected_columns: List[str]


class ChartConfig(BaseModel):
    """Schema for chart configuration."""
    chart_type: str
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    column: Optional[str] = None


class ProfileData(BaseModel):
    """Schema for data profiling results."""
    dataset_overview: Dict[str, Any]
    column_profiles: Dict[str, Dict[str, Any]]
    correlations: Optional[Dict[str, Any]] = None
    memory_usage: Dict[str, Any]


class AnalysisResult(BaseModel):
    """Schema for complete analysis results."""
    dataset_id: UUID
    profile_data: ProfileData
    quality_issues: List[QualityIssue]
    recommendations: List[Recommendation]
    visualizations: List[ChartConfig]
    generated_at: datetime


class AnalysisCacheCreate(BaseModel):
    """Schema for creating analysis cache entries."""
    dataset_id: UUID
    analysis_type: str
    results_json: Dict[str, Any]
    expires_at: Optional[datetime] = None


class AnalysisCache(BaseModel):
    """Schema for analysis cache response."""
    cache_id: UUID
    dataset_id: UUID
    analysis_type: str
    results_json: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class QualityScores(BaseModel):
    """Schema for quality scores."""
    overall: float
    completeness: float
    uniqueness: float
    consistency: float
    validity: float


class QualityIssueSummary(BaseModel):
    """Schema for quality issue summary."""
    total_issues: int
    critical: int
    high: int
    medium: int
    low: int
    info: int


class QualityAssessmentResponse(BaseModel):
    """Schema for quality assessment response."""
    assessment_timestamp: str
    dataset_summary: Dict[str, Any]
    quality_scores: QualityScores
    issues: List[Dict[str, Any]]
    issue_summary: QualityIssueSummary
    prioritized_recommendations: List[Dict[str, Any]]
    column_quality: Dict[str, Dict[str, Any]]