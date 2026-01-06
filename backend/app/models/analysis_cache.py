"""
Analysis cache model definition.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class AnalysisCache(Base):
    """Analysis cache model for storing computed analysis results."""
    
    __tablename__ = "analysis_cache"
    
    cache_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    analysis_type = Column(String(100), nullable=False)
    results_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="analysis_cache")
    
    def __repr__(self):
        return f"<AnalysisCache(cache_id={self.cache_id}, analysis_type={self.analysis_type})>"