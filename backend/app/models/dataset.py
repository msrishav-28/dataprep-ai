"""
Dataset model definition.
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Dataset(Base):
    """Dataset model for storing uploaded file metadata."""
    
    __tablename__ = "datasets"
    
    dataset_id = Column(
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
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    num_rows = Column(Integer, nullable=True)
    num_columns = Column(Integer, nullable=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="uploaded", nullable=False)
    column_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    transformations = relationship("Transformation", back_populates="dataset", cascade="all, delete-orphan")
    analysis_cache = relationship("AnalysisCache", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset(dataset_id={self.dataset_id}, filename={self.filename})>"