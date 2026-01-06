"""
Analysis endpoints for data profiling operations.
"""
from typing import Dict, Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.analysis_service import analysis_service

router = APIRouter()


@router.post("/start/{dataset_id}")
def start_background_analysis(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start background analysis task for a dataset.
    
    Args:
        dataset_id: UUID of the dataset to analyze
        db: Database session
        
    Returns:
        Task information including task ID for status tracking
    """
    return analysis_service.start_background_analysis(db, dataset_id)


@router.get("/task/{task_id}")
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a background analysis task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Task status information including progress and results
    """
    return analysis_service.get_task_status(task_id)


@router.post("/memory/start/{dataset_id}")
def start_memory_analysis(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start background memory analysis task for a dataset.
    
    Args:
        dataset_id: UUID of the dataset to analyze
        db: Database session
        
    Returns:
        Task information including task ID for status tracking
    """
    return analysis_service.start_memory_analysis(db, dataset_id)


@router.get("/profile/{dataset_id}")
def get_dataset_profile(
    dataset_id: UUID,
    force_regenerate: bool = Query(False, description="Force regeneration of profile even if cached"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive data profile for a dataset.
    
    Args:
        dataset_id: UUID of the dataset to profile
        force_regenerate: Whether to force regeneration even if cached
        db: Database session
        
    Returns:
        Comprehensive data profile including statistics, quality assessment, and custom metrics
    """
    return analysis_service.get_or_generate_profile(db, dataset_id, force_regenerate)


@router.get("/memory/{dataset_id}")
def get_memory_analysis(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get memory usage analysis for a dataset.
    
    Args:
        dataset_id: UUID of the dataset to analyze
        db: Database session
        
    Returns:
        Memory usage analysis including total usage and per-column breakdown
    """
    return analysis_service.get_memory_analysis(db, dataset_id)


@router.get("/columns/{dataset_id}")
def get_column_statistics(
    dataset_id: UUID,
    column_name: Optional[str] = Query(None, description="Specific column name to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed statistics for dataset columns.
    
    Args:
        dataset_id: UUID of the dataset
        column_name: Optional specific column name to analyze
        db: Database session
        
    Returns:
        Column statistics including descriptive stats and data type information
    """
    return analysis_service.get_column_statistics(db, dataset_id, column_name)


@router.get("/summary/{dataset_id}")
def get_analysis_summary(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get summary of available analysis results for a dataset.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Summary of available analyses and dataset information
    """
    return analysis_service.get_analysis_summary(db, dataset_id)


@router.delete("/cache/{dataset_id}")
def invalidate_analysis_cache(
    dataset_id: UUID,
    analysis_type: Optional[str] = Query(None, description="Specific analysis type to invalidate"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Invalidate cached analysis results for a dataset.
    
    Args:
        dataset_id: UUID of the dataset
        analysis_type: Optional specific analysis type to invalidate
        db: Database session
        
    Returns:
        Success status and message
    """
    success = analysis_service.invalidate_cache(db, dataset_id, analysis_type)
    
    if success:
        return {
            "success": True,
            "message": f"Cache invalidated for dataset {dataset_id}"
        }
    else:
        return {
            "success": False,
            "message": "No cache entries found to invalidate"
        }


@router.get("/quality/{dataset_id}")
def get_quality_assessment(
    dataset_id: UUID,
    force_regenerate: bool = Query(False, description="Force regeneration even if cached"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive quality assessment for a dataset.
    
    This endpoint performs detailed quality analysis including:
    - Missing value detection with severity levels
    - Duplicate row identification
    - Outlier detection using Z-score and IQR methods
    - Data type inconsistency detection
    - Prioritized recommendations for data cleaning
    
    Args:
        dataset_id: UUID of the dataset to assess
        force_regenerate: Whether to force regeneration even if cached
        db: Database session
        
    Returns:
        Comprehensive quality assessment with scores, issues, and recommendations
    """
    from app.services.quality_service import quality_service
    from app.models.dataset import Dataset
    from app.services.csv_parser import csv_parser
    
    # Get dataset from database
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Check cache first (unless force_regenerate)
    if not force_regenerate:
        cached = analysis_service._get_cached_analysis(db, dataset_id, "quality_assessment")
        if cached:
            return cached
    
    # Load dataset
    try:
        df = csv_parser.parse_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
    
    # Perform quality assessment
    assessment = quality_service.assess_quality(df)
    
    # Cache results
    analysis_service._cache_analysis_results(db, dataset_id, "quality_assessment", assessment)
    
    return assessment


@router.get("/visualizations/{dataset_id}")
def get_visualizations(
    dataset_id: UUID,
    force_regenerate: bool = Query(False, description="Force regeneration even if cached"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get interactive visualizations for a dataset.
    
    This endpoint generates appropriate visualizations including:
    - Distribution plots (histograms) for numerical columns
    - Box plots for outlier visualization
    - Count plots (bar charts) for categorical variables
    - Correlation heatmap for numerical variables
    - Missing value heatmap showing patterns of missingness
    
    Args:
        dataset_id: UUID of the dataset to visualize
        force_regenerate: Whether to force regeneration even if cached
        db: Database session
        
    Returns:
        Dict containing chart configurations for Plotly rendering
    """
    from app.services.visualization_service import visualization_service
    from app.models.dataset import Dataset
    from app.services.csv_parser import csv_parser
    
    # Get dataset from database
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Check cache first (unless force_regenerate)
    if not force_regenerate:
        cached = analysis_service._get_cached_analysis(db, dataset_id, "visualizations")
        if cached:
            return cached
    
    # Load dataset
    try:
        df = csv_parser.parse_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
    
    # Generate visualizations
    visualizations = visualization_service.generate_all_visualizations(df, dataset.filename)
    
    # Cache results
    analysis_service._cache_analysis_results(db, dataset_id, "visualizations", visualizations)
    
    return visualizations

