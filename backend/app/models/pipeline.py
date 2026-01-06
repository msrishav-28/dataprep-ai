"""
Pipeline model for storing reusable preprocessing workflows.
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Pipeline(Base):
    """Model for storing reusable preprocessing pipelines."""
    
    __tablename__ = "pipelines"
    
    pipeline_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.dataset_id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Pipeline metadata
    pipeline_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)  # e.g., "cleaning", "encoding", "complete"
    
    # Transformation steps stored as JSONB
    transformations_json = Column(JSONB, nullable=False, default=list)
    
    # Pipeline configuration
    config_json = Column(JSONB, nullable=True)  # Additional settings
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_modified = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Sharing and usage
    is_public = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Tags for categorization
    tags = Column(JSONB, nullable=True, default=list)
    
    # Relationships
    user = relationship("User", backref="pipelines")
    dataset = relationship("Dataset", backref="pipelines")
    
    def __repr__(self):
        return f"<Pipeline(pipeline_id={self.pipeline_id}, name={self.pipeline_name})>"
