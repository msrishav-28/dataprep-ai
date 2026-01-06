"""
Analysis service for managing data profiling and caching.
"""
import pandas as pd
from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
import logging
from datetime import datetime, timedelta
import json

from app.models.analysis_cache import AnalysisCache
from app.models.dataset import Dataset
from app.services.profiling_service import profiling_service
from app.services.csv_parser import csv_parser
from app.services.tasks import analyze_dataset_task, analyze_memory_usage_task, get_task_status

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for managing data analysis operations and caching."""
    
    def __init__(self):
        """Initialize the analysis service."""
        self.logger = logging.getLogger(__name__)
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
    
    def start_background_analysis(self, db: Session, dataset_id: UUID) -> Dict[str, Any]:
        """
        Start background analysis task for a dataset.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to analyze
            
        Returns:
            Dict containing task information
            
        Raises:
            HTTPException: If dataset not found
        """
        try:
            # Verify dataset exists
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )
            
            # Update dataset status to processing
            dataset.status = "analyzing"
            db.commit()
            
            # Start background task
            task = analyze_dataset_task.delay(str(dataset_id))
            
            self.logger.info(f"Started background analysis task {task.id} for dataset {dataset_id}")
            
            return {
                "task_id": task.id,
                "dataset_id": str(dataset_id),
                "status": "started",
                "message": "Background analysis task started"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error starting background analysis for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start background analysis: {str(e)}"
            )
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a background analysis task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Dict containing task status information
        """
        try:
            return get_task_status(task_id)
        except Exception as e:
            self.logger.error(f"Error getting task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "state": "ERROR",
                "error": str(e)
            }
    
    def start_memory_analysis(self, db: Session, dataset_id: UUID) -> Dict[str, Any]:
        """
        Start background memory analysis task for a dataset.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to analyze
            
        Returns:
            Dict containing task information
        """
        try:
            # Verify dataset exists
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )
            
            # Start background task
            task = analyze_memory_usage_task.delay(str(dataset_id))
            
            self.logger.info(f"Started memory analysis task {task.id} for dataset {dataset_id}")
            
            return {
                "task_id": task.id,
                "dataset_id": str(dataset_id),
                "status": "started",
                "message": "Memory analysis task started"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error starting memory analysis for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start memory analysis: {str(e)}"
            )
        """
        Get cached profile or generate new one if not available.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to profile
            force_regenerate: Whether to force regeneration even if cached
            
        Returns:
            Dict containing profiling results
            
        Raises:
            HTTPException: If dataset not found or profiling fails
        """
        try:
            # Get dataset
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )
            
            # Check for cached results if not forcing regeneration
            if not force_regenerate:
                cached_profile = self._get_cached_analysis(db, dataset_id, "comprehensive_profile")
                if cached_profile:
                    self.logger.info(f"Using cached profile for dataset {dataset_id}")
                    return cached_profile
            
            # Load dataset for profiling
            df = self._load_dataset(dataset.file_path)
            if df is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to load dataset for profiling"
                )
            
            # Generate comprehensive profile
            self.logger.info(f"Generating comprehensive profile for dataset {dataset_id}")
            profile_results = profiling_service.generate_comprehensive_profile(df, dataset.filename)
            
            # Cache the results
            self._cache_analysis_results(db, dataset_id, "comprehensive_profile", profile_results)
            
            # Update dataset status
            dataset.status = "analyzed"
            db.commit()
            
            return profile_results
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error generating profile for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate data profile: {str(e)}"
            )
    
    def get_memory_analysis(self, db: Session, dataset_id: UUID) -> Dict[str, Any]:
        """
        Get memory usage analysis for a dataset.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset to analyze
            
        Returns:
            Dict containing memory analysis results
        """
        try:
            # Check for cached results
            cached_memory = self._get_cached_analysis(db, dataset_id, "memory_analysis")
            if cached_memory:
                return cached_memory
            
            # Get dataset
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )
            
            # Load dataset for memory analysis
            df = self._load_dataset(dataset.file_path)
            if df is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to load dataset for memory analysis"
                )
            
            # Calculate memory usage
            memory_results = profiling_service.calculate_memory_usage(df)
            
            # Cache the results
            self._cache_analysis_results(db, dataset_id, "memory_analysis", memory_results)
            
            return memory_results
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error analyzing memory for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze memory usage: {str(e)}"
            )
    
    def get_column_statistics(self, db: Session, dataset_id: UUID, column_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed statistics for specific column(s).
        
        Args:
            db: Database session
            dataset_id: ID of the dataset
            column_name: Optional specific column name
            
        Returns:
            Dict containing column statistics
        """
        try:
            # Get comprehensive profile (cached or generated)
            profile = self.get_or_generate_profile(db, dataset_id)
            
            column_stats = profile.get("column_statistics", {})
            
            if column_name:
                if column_name not in column_stats:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Column '{column_name}' not found in dataset"
                    )
                return {column_name: column_stats[column_name]}
            
            return column_stats
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting column statistics for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get column statistics: {str(e)}"
            )
    
    def _get_cached_analysis(self, db: Session, dataset_id: UUID, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis results if available and not expired.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset
            analysis_type: Type of analysis to retrieve
            
        Returns:
            Cached results if available, None otherwise
        """
        try:
            cached_result = db.query(AnalysisCache).filter(
                AnalysisCache.dataset_id == dataset_id,
                AnalysisCache.analysis_type == analysis_type
            ).first()
            
            if cached_result:
                # Check if cache has expired
                if cached_result.expires_at and cached_result.expires_at < datetime.utcnow():
                    # Delete expired cache
                    db.delete(cached_result)
                    db.commit()
                    return None
                
                return cached_result.results_json
            
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error retrieving cached analysis: {e}")
            return None
    
    def _cache_analysis_results(self, db: Session, dataset_id: UUID, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Cache analysis results in the database.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset
            analysis_type: Type of analysis being cached
            results: Results to cache
        """
        try:
            # Delete existing cache for this dataset and analysis type
            existing_cache = db.query(AnalysisCache).filter(
                AnalysisCache.dataset_id == dataset_id,
                AnalysisCache.analysis_type == analysis_type
            ).first()
            
            if existing_cache:
                db.delete(existing_cache)
            
            # Create new cache entry
            cache_entry = AnalysisCache(
                dataset_id=dataset_id,
                analysis_type=analysis_type,
                results_json=results,
                expires_at=datetime.utcnow() + timedelta(hours=self.cache_expiry_hours)
            )
            
            db.add(cache_entry)
            db.commit()
            
            self.logger.info(f"Cached {analysis_type} results for dataset {dataset_id}")
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Database error caching analysis results: {e}")
    
    def _load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load dataset from file path.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Use the existing CSV parser service
            return csv_parser.parse_csv_file(file_path)
            
        except Exception as e:
            self.logger.error(f"Error loading dataset from {file_path}: {e}")
            return None
    
    def invalidate_cache(self, db: Session, dataset_id: UUID, analysis_type: Optional[str] = None) -> bool:
        """
        Invalidate cached analysis results.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset
            analysis_type: Optional specific analysis type to invalidate
            
        Returns:
            True if cache was invalidated, False otherwise
        """
        try:
            query = db.query(AnalysisCache).filter(AnalysisCache.dataset_id == dataset_id)
            
            if analysis_type:
                query = query.filter(AnalysisCache.analysis_type == analysis_type)
            
            deleted_count = query.delete()
            db.commit()
            
            self.logger.info(f"Invalidated {deleted_count} cache entries for dataset {dataset_id}")
            return deleted_count > 0
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Database error invalidating cache: {e}")
            return False
    
    def get_analysis_summary(self, db: Session, dataset_id: UUID) -> Dict[str, Any]:
        """
        Get a summary of available analysis results.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset
            
        Returns:
            Dict containing analysis summary
        """
        try:
            # Get dataset info
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )
            
            # Get cached analysis types
            cached_analyses = db.query(AnalysisCache).filter(
                AnalysisCache.dataset_id == dataset_id
            ).all()
            
            summary = {
                "dataset_id": str(dataset_id),
                "dataset_name": dataset.filename,
                "dataset_status": dataset.status,
                "available_analyses": [
                    {
                        "type": cache.analysis_type,
                        "generated_at": cache.created_at.isoformat(),
                        "expires_at": cache.expires_at.isoformat() if cache.expires_at else None
                    }
                    for cache in cached_analyses
                ],
                "dataset_info": {
                    "num_rows": dataset.num_rows,
                    "num_columns": dataset.num_columns,
                    "file_size_bytes": dataset.file_size_bytes,
                    "upload_date": dataset.upload_date.isoformat() if dataset.upload_date else None
                }
            }
            
            return summary
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting analysis summary for dataset {dataset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get analysis summary: {str(e)}"
            )


# Global instance
analysis_service = AnalysisService()