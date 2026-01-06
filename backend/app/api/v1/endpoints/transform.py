"""
Transformation API endpoints for data preprocessing operations.
"""
from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.services.transformation_service import (
    transformation_service, 
    TransformationType, 
    TransformationParams
)
from app.services.csv_parser import csv_parser
from app.models.dataset import Dataset

router = APIRouter()


class TransformationRequest(BaseModel):
    """Request body for transformation operations."""
    transformation_type: str
    columns: List[str]
    constant_value: Optional[Any] = None
    threshold: float = 3.0
    iqr_multiplier: float = 1.5
    target_dtype: Optional[str] = None
    filter_condition: Optional[str] = None
    new_name: Optional[str] = None


@router.post("/preview/{dataset_id}")
def preview_transformation(
    dataset_id: UUID,
    request: TransformationRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Preview the effect of a transformation before applying.
    
    This endpoint generates a preview showing:
    - Sample of data before and after transformation
    - Statistics comparison (before/after)
    - Number of affected rows
    - Warnings or potential issues
    - Explanation of the transformation
    
    Args:
        dataset_id: UUID of the dataset
        request: Transformation parameters
        db: Database session
        
    Returns:
        TransformationPreview with before/after comparison
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Load dataset
    try:
        df = csv_parser.parse_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
    
    # Validate transformation type
    try:
        transform_type = TransformationType(request.transformation_type)
    except ValueError:
        valid_types = [t.value for t in TransformationType]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid transformation type. Valid types: {valid_types}"
        )
    
    # Create parameters
    params = TransformationParams(
        columns=request.columns,
        constant_value=request.constant_value,
        threshold=request.threshold,
        iqr_multiplier=request.iqr_multiplier,
        target_dtype=request.target_dtype,
        filter_condition=request.filter_condition,
        new_name=request.new_name
    )
    
    # Generate preview
    preview = transformation_service.preview_transformation(df, transform_type, params)
    
    return preview.to_dict()


@router.post("/apply/{dataset_id}")
def apply_transformation(
    dataset_id: UUID,
    request: TransformationRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Apply a transformation to the dataset.
    
    This endpoint applies the specified transformation and:
    - Updates the dataset
    - Records the transformation in history
    - Returns before/after statistics
    
    Args:
        dataset_id: UUID of the dataset
        request: Transformation parameters
        db: Database session
        
    Returns:
        TransformationResult with success status and statistics
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Load dataset
    try:
        df = csv_parser.parse_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
    
    # Validate transformation type
    try:
        transform_type = TransformationType(request.transformation_type)
    except ValueError:
        valid_types = [t.value for t in TransformationType]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid transformation type. Valid types: {valid_types}"
        )
    
    # Create parameters
    params = TransformationParams(
        columns=request.columns,
        constant_value=request.constant_value,
        threshold=request.threshold,
        iqr_multiplier=request.iqr_multiplier,
        target_dtype=request.target_dtype,
        filter_condition=request.filter_condition,
        new_name=request.new_name
    )
    
    # Apply transformation
    df_transformed, result = transformation_service.apply_transformation(
        df, transform_type, params, str(dataset_id)
    )
    
    # Save transformed dataset back
    if result.success:
        try:
            df_transformed.to_csv(dataset.file_path, index=False)
            
            # Update dataset metadata
            dataset.num_rows = len(df_transformed)
            dataset.num_columns = len(df_transformed.columns)
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error saving transformed dataset: {str(e)}")
    
    return result.to_dict()


@router.get("/history/{dataset_id}")
def get_transformation_history(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the transformation history for a dataset.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        List of applied transformations
    """
    history = transformation_service.get_transformation_history(str(dataset_id))
    
    return {
        "dataset_id": str(dataset_id),
        "transformations": history,
        "total_count": len(history)
    }


@router.post("/undo/{dataset_id}")
def undo_transformation(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Undo the last transformation applied to a dataset.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Success status and message
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Try to undo
    previous_df = transformation_service.undo_transformation(str(dataset_id))
    
    if previous_df is not None:
        try:
            previous_df.to_csv(dataset.file_path, index=False)
            
            # Update dataset metadata
            dataset.num_rows = len(previous_df)
            dataset.num_columns = len(previous_df.columns)
            db.commit()
            
            return {
                "success": True,
                "message": "Successfully reverted to previous state",
                "rows": len(previous_df),
                "columns": len(previous_df.columns)
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error reverting dataset: {str(e)}")
    else:
        return {
            "success": False,
            "message": "No previous state available to revert to"
        }


@router.get("/types")
def get_transformation_types() -> Dict[str, Any]:
    """
    Get available transformation types with descriptions.
    
    Returns:
        List of available transformation types and their categories
    """
    return {
        "imputation": {
            "description": "Handle missing values",
            "types": [
                {"value": "impute_mean", "label": "Mean Imputation", "description": "Replace with column mean"},
                {"value": "impute_median", "label": "Median Imputation", "description": "Replace with column median"},
                {"value": "impute_mode", "label": "Mode Imputation", "description": "Replace with most frequent value"},
                {"value": "impute_constant", "label": "Constant Value", "description": "Replace with specified value"},
                {"value": "impute_forward_fill", "label": "Forward Fill", "description": "Use previous valid value"},
                {"value": "impute_backward_fill", "label": "Backward Fill", "description": "Use next valid value"}
            ]
        },
        "outlier_treatment": {
            "description": "Handle outliers in numerical data",
            "types": [
                {"value": "remove_outliers_zscore", "label": "Remove (Z-score)", "description": "Remove rows with Z-score > threshold"},
                {"value": "remove_outliers_iqr", "label": "Remove (IQR)", "description": "Remove rows outside IQR bounds"},
                {"value": "cap_outliers_zscore", "label": "Cap (Z-score)", "description": "Cap values at Z-score threshold"},
                {"value": "cap_outliers_iqr", "label": "Cap (IQR)", "description": "Cap values at IQR bounds"}
            ]
        },
        "encoding": {
            "description": "Convert categorical variables",
            "types": [
                {"value": "encode_onehot", "label": "One-Hot Encoding", "description": "Create binary columns for categories"},
                {"value": "encode_label", "label": "Label Encoding", "description": "Convert to integer codes"}
            ]
        },
        "scaling": {
            "description": "Normalize numerical features",
            "types": [
                {"value": "scale_standard", "label": "Standard Scaling", "description": "Scale to mean=0, std=1"},
                {"value": "scale_minmax", "label": "Min-Max Scaling", "description": "Scale to [0, 1] range"}
            ]
        },
        "data_management": {
            "description": "Modify dataset structure",
            "types": [
                {"value": "drop_column", "label": "Drop Column", "description": "Remove specified columns"},
                {"value": "remove_duplicates", "label": "Remove Duplicates", "description": "Remove duplicate rows"},
                {"value": "rename_column", "label": "Rename Column", "description": "Rename a column"},
                {"value": "convert_dtype", "label": "Convert Type", "description": "Change column data type"}
            ]
        }
    }
