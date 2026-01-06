"""
Transformation model definition.
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Transformation(Base):
    """Transformation model for storing applied data transformations."""
    
    __tablename__ = "transformations"
    
    transformation_id = Column(
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
    operation_type = Column(String(100), nullable=False)
    parameters = Column(JSONB, nullable=False)
    applied_at = Column(DateTime(timezone=True), server_default=func.now())
    sequence_order = Column(Integer, nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="transformations")
    
    def __repr__(self):
        return f"<Transformation(transformation_id={self.transformation_id}, operation_type={self.operation_type})>"