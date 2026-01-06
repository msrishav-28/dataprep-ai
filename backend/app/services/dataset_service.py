"""
Dataset service for database operations related to datasets.
"""
from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
import logging

from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.models.user import User

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for dataset database operations."""
    
    @staticmethod
    def create_dataset(db: Session, dataset_data: DatasetCreate, user_id: UUID) -> Dataset:
        """
        Create a new dataset record in the database.
        
        Args:
            db: Database session
            dataset_data: Dataset creation data
            user_id: ID of the user creating the dataset
            
        Returns:
            Dataset: Created dataset instance
            
        Raises:
            HTTPException: If creation fails
        """
        try:
            # Verify user exists
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Create dataset instance
            db_dataset = Dataset(
                user_id=user_id,
                filename=dataset_data.filename,
                file_path=dataset_data.file_path,
                file_size_bytes=dataset_data.file_size_bytes,
                num_rows=dataset_data.num_rows,
                num_columns=dataset_data.num_columns,
                column_metadata=dataset_data.column_metadata,
                status="uploaded"
            )
            
            db.add(db_dataset)
            db.commit()
            db.refresh(db_dataset)
            
            logger.info(f"Created dataset: {db_dataset.dataset_id}")
            return db_dataset
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error creating dataset: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create dataset record"
            )
    
    @staticmethod
    def get_dataset(db: Session, dataset_id: UUID, user_id: Optional[UUID] = None) -> Optional[Dataset]:
        """
        Retrieve a dataset by ID.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to retrieve
            user_id: Optional user ID for ownership verification
            
        Returns:
            Dataset: Dataset instance if found, None otherwise
        """
        try:
            query = db.query(Dataset).filter(Dataset.dataset_id == dataset_id)
            
            if user_id:
                query = query.filter(Dataset.user_id == user_id)
            
            return query.first()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving dataset {dataset_id}: {e}")
            return None
    
    @staticmethod
    def get_user_datasets(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """
        Retrieve datasets for a specific user.
        
        Args:
            db: Database session
            user_id: ID of the user
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Dataset]: List of user's datasets
        """
        try:
            return db.query(Dataset).filter(Dataset.user_id == user_id).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving user datasets: {e}")
            return []
    
    @staticmethod
    def update_dataset(db: Session, dataset_id: UUID, dataset_update: DatasetUpdate, user_id: Optional[UUID] = None) -> Optional[Dataset]:
        """
        Update a dataset record.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to update
            dataset_update: Update data
            user_id: Optional user ID for ownership verification
            
        Returns:
            Dataset: Updated dataset instance if successful, None otherwise
        """
        try:
            query = db.query(Dataset).filter(Dataset.dataset_id == dataset_id)
            
            if user_id:
                query = query.filter(Dataset.user_id == user_id)
            
            db_dataset = query.first()
            if not db_dataset:
                return None
            
            # Update fields
            update_data = dataset_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_dataset, field, value)
            
            db.commit()
            db.refresh(db_dataset)
            
            logger.info(f"Updated dataset: {dataset_id}")
            return db_dataset
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error updating dataset {dataset_id}: {e}")
            return None
    
    @staticmethod
    def delete_dataset(db: Session, dataset_id: UUID, user_id: Optional[UUID] = None) -> bool:
        """
        Delete a dataset record.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to delete
            user_id: Optional user ID for ownership verification
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            query = db.query(Dataset).filter(Dataset.dataset_id == dataset_id)
            
            if user_id:
                query = query.filter(Dataset.user_id == user_id)
            
            db_dataset = query.first()
            if not db_dataset:
                return False
            
            db.delete(db_dataset)
            db.commit()
            
            logger.info(f"Deleted dataset: {dataset_id}")
            return True
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error deleting dataset {dataset_id}: {e}")
            return False


# Global instance
dataset_service = DatasetService()