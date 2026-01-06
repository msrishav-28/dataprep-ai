"""
Dataset API endpoints for file upload and management.
"""
import io
import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.database import get_db
from app.core.database_utils import get_dummy_user_id
from app.schemas.dataset import Dataset, DatasetCreate, DatasetPreview
from app.services.file_storage import file_storage
from app.services.file_validation import file_validator
from app.services.dataset_service import dataset_service
from app.services.csv_parser import csv_parser

router = APIRouter()


@router.post("/upload", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a CSV dataset file.
    
    This endpoint handles file upload with comprehensive validation:
    - File format validation (CSV only)
    - File size validation (max 1GB)
    - Encoding detection and validation
    - Secure file storage
    - Database record creation
    
    Args:
        file: Uploaded CSV file
        db: Database session
        
    Returns:
        Dataset: Created dataset information
        
    Raises:
        HTTPException: If validation fails or upload processing fails
    """
    # Validate file format
    file_validator.validate_file_format(file.filename)
    
    # Read file content into memory
    file_content = await file.read()
    file_size = len(file_content)
    
    # Validate file size
    file_validator.validate_file_size(file_size)
    
    # Create file-like object for processing
    file_stream = io.BytesIO(file_content)
    
    # Detect and validate encoding
    encoding, confidence = file_validator.detect_encoding(file_stream)
    
    # Validate CSV content
    file_validator.validate_csv_content(file_stream, encoding)
    
    try:
        # Parse CSV and extract metadata
        df, metadata = csv_parser.parse_csv(file_stream, encoding, file.filename)
        
        # Store file in MinIO/S3
        file_path = file_storage.store_file(
            file_content=file_stream,
            filename=file.filename,
            content_type=file.content_type or "text/csv"
        )
        
        # Create dataset record in database with metadata
        # Note: For MVP, we'll use a dummy user_id. In production, this would come from authentication
        dummy_user_id = get_dummy_user_id()
        
        dataset_data = DatasetCreate(
            filename=file.filename,
            file_path=file_path,
            file_size_bytes=file_size,
            num_rows=metadata.get('num_rows'),
            num_columns=metadata.get('num_columns'),
            column_metadata=metadata
        )
        
        dataset = dataset_service.create_dataset(
            db=db,
            dataset_data=dataset_data,
            user_id=dummy_user_id
        )
        
        return dataset
        
    except Exception as e:
        # Clean up stored file if database operation fails
        try:
            if 'file_path' in locals():
                file_storage.delete_file(file_path)
        except Exception:
            pass  # Log but don't fail the error response
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file upload: {str(e)}"
        )


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Retrieve dataset information by ID.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Dataset: Dataset information
        
    Raises:
        HTTPException: If dataset not found
    """
    dataset = dataset_service.get_dataset(db=db, dataset_id=dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def get_dataset_preview(
    dataset_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a preview of the dataset with sample data.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        DatasetPreview: Dataset preview with sample data
        
    Raises:
        HTTPException: If dataset not found or file cannot be read
    """
    dataset = dataset_service.get_dataset(db=db, dataset_id=dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        # Retrieve file from storage
        file_content = file_storage.get_file(dataset.file_path)
        
        # Detect encoding (should be cached in metadata, but detect as fallback)
        encoding = 'utf-8'  # Default encoding
        if dataset.column_metadata and 'encoding' in dataset.column_metadata:
            encoding = dataset.column_metadata['encoding']
        else:
            encoding, _ = file_validator.detect_encoding(file_content)
        
        # Parse CSV to get sample data
        df, _ = csv_parser.parse_csv(file_content, encoding, dataset.filename)
        sample_data = csv_parser.get_sample_data(df)
        
        return DatasetPreview(
            dataset_id=dataset.dataset_id,
            filename=dataset.filename,
            num_rows=dataset.num_rows,
            num_columns=dataset.num_columns,
            sample_data=sample_data,
            column_info=dataset.column_metadata
        )
        
    except Exception as e:
        logger.error(f"Error generating dataset preview for {dataset_id}: {e}")
        # Return basic preview without sample data if parsing fails
        return DatasetPreview(
            dataset_id=dataset.dataset_id,
            filename=dataset.filename,
            num_rows=dataset.num_rows,
            num_columns=dataset.num_columns,
            sample_data={'error': f"Failed to load sample data: {str(e)}"},
            column_info=dataset.column_metadata
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a dataset and its associated file.
    
    Args:
        dataset_id: UUID of the dataset to delete
        db: Database session
        
    Raises:
        HTTPException: If dataset not found or deletion fails
    """
    # Get dataset to retrieve file path
    dataset = dataset_service.get_dataset(db=db, dataset_id=dataset_id)
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    try:
        # Delete file from storage
        file_storage.delete_file(dataset.file_path)
        
        # Delete database record
        success = dataset_service.delete_dataset(db=db, dataset_id=dataset_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset record"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@router.get("/", response_model=List[Dataset])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List datasets for the current user.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List[Dataset]: List of datasets
    """
    # Note: For MVP, we'll use a dummy user_id. In production, this would come from authentication
    dummy_user_id = get_dummy_user_id()
    
    datasets = dataset_service.get_user_datasets(
        db=db,
        user_id=dummy_user_id,
        skip=skip,
        limit=limit
    )
    
    return datasets