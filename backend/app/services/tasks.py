"""
Celery background tasks for data processing operations.
"""
import pandas as pd
from typing import Dict, Any, Optional
from uuid import UUID
import logging
from datetime import datetime
import traceback

from celery import current_task
from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.analysis_cache import AnalysisCache
from app.services.profiling_service import profiling_service
from app.services.csv_parser import csv_parser

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analyze_dataset")
def analyze_dataset_task(self, dataset_id: str) -> Dict[str, Any]:
    """
    Background task to perform comprehensive dataset analysis.
    
    Args:
        dataset_id: UUID string of the dataset to analyze
        
    Returns:
        Dict containing analysis results and task status
    """
    task_id = self.request.id
    logger.info(f"Starting dataset analysis task {task_id} for dataset {dataset_id}")
    
    # Update task state to PROGRESS
    self.update_state(
        state='PROGRESS',
        meta={
            'current_step': 'initializing',
            'total_steps': 5,
            'step': 1,
            'message': 'Initializing analysis...'
        }
    )
    
    db = SessionLocal()
    try:
        # Get dataset from database
        dataset = db.query(Dataset).filter(Dataset.dataset_id == UUID(dataset_id)).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'loading_data',
                'total_steps': 5,
                'step': 2,
                'message': f'Loading dataset: {dataset.filename}'
            }
        )
        
        # Load dataset
        df = csv_parser.parse_csv_file(dataset.file_path)
        if df is None:
            raise ValueError(f"Failed to load dataset from {dataset.file_path}")
        
        logger.info(f"Loaded dataset {dataset_id} with shape {df.shape}")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'profiling',
                'total_steps': 5,
                'step': 3,
                'message': 'Generating comprehensive profile...'
            }
        )
        
        # Generate comprehensive profile
        profile_results = profiling_service.generate_comprehensive_profile(df, dataset.filename)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'caching',
                'total_steps': 5,
                'step': 4,
                'message': 'Caching analysis results...'
            }
        )
        
        # Cache the results
        _cache_analysis_results(db, UUID(dataset_id), "comprehensive_profile", profile_results)
        
        # Update dataset status
        dataset.status = "analyzed"
        db.commit()
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'complete',
                'total_steps': 5,
                'step': 5,
                'message': 'Analysis complete!'
            }
        )
        
        logger.info(f"Completed dataset analysis task {task_id} for dataset {dataset_id}")
        
        return {
            'status': 'SUCCESS',
            'dataset_id': dataset_id,
            'task_id': task_id,
            'profile_summary': {
                'num_rows': profile_results['dataset_overview']['num_rows'],
                'num_columns': profile_results['dataset_overview']['num_columns'],
                'memory_usage_mb': profile_results['dataset_overview']['memory_usage_mb'],
                'quality_score': profile_results['data_quality_summary']['overall_score']
            },
            'completed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in dataset analysis task {task_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update dataset status to error
        try:
            dataset = db.query(Dataset).filter(Dataset.dataset_id == UUID(dataset_id)).first()
            if dataset:
                dataset.status = "analysis_failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update dataset status: {db_error}")
        
        # Update task state to FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'dataset_id': dataset_id,
                'task_id': task_id
            }
        )
        
        raise e
        
    finally:
        db.close()


@celery_app.task(bind=True, name="analyze_memory_usage")
def analyze_memory_usage_task(self, dataset_id: str) -> Dict[str, Any]:
    """
    Background task to analyze dataset memory usage.
    
    Args:
        dataset_id: UUID string of the dataset to analyze
        
    Returns:
        Dict containing memory analysis results
    """
    task_id = self.request.id
    logger.info(f"Starting memory analysis task {task_id} for dataset {dataset_id}")
    
    self.update_state(
        state='PROGRESS',
        meta={
            'current_step': 'loading',
            'message': 'Loading dataset for memory analysis...'
        }
    )
    
    db = SessionLocal()
    try:
        # Get dataset from database
        dataset = db.query(Dataset).filter(Dataset.dataset_id == UUID(dataset_id)).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load dataset
        df = csv_parser.parse_csv_file(dataset.file_path)
        if df is None:
            raise ValueError(f"Failed to load dataset from {dataset.file_path}")
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current_step': 'analyzing',
                'message': 'Calculating memory usage...'
            }
        )
        
        # Calculate memory usage
        memory_results = profiling_service.calculate_memory_usage(df)
        
        # Cache the results
        _cache_analysis_results(db, UUID(dataset_id), "memory_analysis", memory_results)
        
        logger.info(f"Completed memory analysis task {task_id} for dataset {dataset_id}")
        
        return {
            'status': 'SUCCESS',
            'dataset_id': dataset_id,
            'task_id': task_id,
            'memory_results': memory_results,
            'completed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in memory analysis task {task_id}: {e}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'dataset_id': dataset_id,
                'task_id': task_id
            }
        )
        
        raise e
        
    finally:
        db.close()


@celery_app.task(bind=True, name="batch_analyze_datasets")
def batch_analyze_datasets_task(self, dataset_ids: list) -> Dict[str, Any]:
    """
    Background task to analyze multiple datasets in batch.
    
    Args:
        dataset_ids: List of dataset UUID strings to analyze
        
    Returns:
        Dict containing batch analysis results
    """
    task_id = self.request.id
    logger.info(f"Starting batch analysis task {task_id} for {len(dataset_ids)} datasets")
    
    results = {
        'task_id': task_id,
        'total_datasets': len(dataset_ids),
        'successful': [],
        'failed': [],
        'started_at': datetime.utcnow().isoformat()
    }
    
    for i, dataset_id in enumerate(dataset_ids):
        try:
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'current_dataset': i + 1,
                    'total_datasets': len(dataset_ids),
                    'dataset_id': dataset_id,
                    'message': f'Analyzing dataset {i + 1} of {len(dataset_ids)}'
                }
            )
            
            # Analyze individual dataset
            result = analyze_dataset_task.apply(args=[dataset_id])
            
            if result.successful():
                results['successful'].append({
                    'dataset_id': dataset_id,
                    'result': result.result
                })
            else:
                results['failed'].append({
                    'dataset_id': dataset_id,
                    'error': str(result.result)
                })
                
        except Exception as e:
            logger.error(f"Error analyzing dataset {dataset_id} in batch: {e}")
            results['failed'].append({
                'dataset_id': dataset_id,
                'error': str(e)
            })
    
    results['completed_at'] = datetime.utcnow().isoformat()
    results['success_rate'] = len(results['successful']) / len(dataset_ids) * 100
    
    logger.info(f"Completed batch analysis task {task_id}: {len(results['successful'])} successful, {len(results['failed'])} failed")
    
    return results


def _cache_analysis_results(db, dataset_id: UUID, analysis_type: str, results: Dict[str, Any]) -> None:
    """
    Helper function to cache analysis results in the database.
    
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
        from datetime import timedelta
        cache_entry = AnalysisCache(
            dataset_id=dataset_id,
            analysis_type=analysis_type,
            results_json=results,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
        db.add(cache_entry)
        db.commit()
        
        logger.info(f"Cached {analysis_type} results for dataset {dataset_id}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error caching analysis results: {e}")


# Task status monitoring functions
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a background task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Dict containing task status information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            return {
                'task_id': task_id,
                'state': 'PENDING',
                'message': 'Task is waiting to be processed'
            }
        elif result.state == 'PROGRESS':
            return {
                'task_id': task_id,
                'state': 'PROGRESS',
                'current_step': result.info.get('current_step', 'unknown'),
                'step': result.info.get('step', 0),
                'total_steps': result.info.get('total_steps', 0),
                'message': result.info.get('message', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            return {
                'task_id': task_id,
                'state': 'SUCCESS',
                'result': result.result
            }
        elif result.state == 'FAILURE':
            return {
                'task_id': task_id,
                'state': 'FAILURE',
                'error': str(result.info.get('error', 'Unknown error')),
                'traceback': result.info.get('traceback')
            }
        else:
            return {
                'task_id': task_id,
                'state': result.state,
                'info': result.info
            }
            
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return {
            'task_id': task_id,
            'state': 'ERROR',
            'error': str(e)
        }


def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a background task.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Dict containing cancellation status
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {
            'task_id': task_id,
            'status': 'cancelled',
            'message': 'Task has been cancelled'
        }
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        }