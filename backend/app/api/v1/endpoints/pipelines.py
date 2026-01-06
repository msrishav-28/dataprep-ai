"""
Pipeline API endpoints for saving, loading, and managing preprocessing pipelines.
"""
from typing import Any, List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user, get_current_active_user
from app.models.user import User
from app.models.pipeline import Pipeline
from app.models.dataset import Dataset
from app.services.transformation_service import transformation_service
from app.services.csv_parser import csv_parser

router = APIRouter()


# Pydantic schemas
class PipelineCreate(BaseModel):
    """Schema for creating a new pipeline."""
    pipeline_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    transformations: List[dict] = Field(default_factory=list)
    config: Optional[dict] = None
    is_public: bool = False
    tags: List[str] = Field(default_factory=list)


class PipelineUpdate(BaseModel):
    """Schema for updating a pipeline."""
    pipeline_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    transformations: Optional[List[dict]] = None
    config: Optional[dict] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    """Schema for pipeline API responses."""
    pipeline_id: UUID
    user_id: UUID
    dataset_id: Optional[UUID] = None
    pipeline_name: str
    description: Optional[str] = None
    category: Optional[str] = None
    transformations: List[dict]
    config: Optional[dict] = None
    created_at: datetime
    last_modified: datetime
    is_public: bool
    usage_count: int
    tags: List[str]
    
    class Config:
        from_attributes = True


class PipelineListResponse(BaseModel):
    """Schema for listing pipelines."""
    pipelines: List[PipelineResponse]
    total: int
    page: int
    page_size: int


@router.post("/", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    pipeline_data: PipelineCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a new preprocessing pipeline.
    
    Args:
        pipeline_data: Pipeline creation data
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Created pipeline
    """
    pipeline = Pipeline(
        user_id=current_user.user_id,
        pipeline_name=pipeline_data.pipeline_name,
        description=pipeline_data.description,
        category=pipeline_data.category,
        transformations_json=pipeline_data.transformations,
        config_json=pipeline_data.config,
        is_public=pipeline_data.is_public,
        tags=pipeline_data.tags
    )
    
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    
    return _pipeline_to_response(pipeline)


@router.post("/from-dataset/{dataset_id}", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline_from_dataset(
    dataset_id: UUID,
    pipeline_name: str = Query(..., min_length=1),
    description: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a pipeline from the transformation history of a dataset.
    
    Args:
        dataset_id: UUID of the source dataset
        pipeline_name: Name for the new pipeline
        description: Optional description
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Created pipeline
    """
    # Verify dataset belongs to user
    dataset = db.query(Dataset).filter(
        Dataset.dataset_id == dataset_id,
        Dataset.user_id == current_user.user_id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Get transformation history
    history = transformation_service.get_transformation_history(str(dataset_id))
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transformations applied to this dataset yet"
        )
    
    pipeline = Pipeline(
        user_id=current_user.user_id,
        dataset_id=dataset_id,
        pipeline_name=pipeline_name,
        description=description or f"Pipeline created from {dataset.filename}",
        transformations_json=history,
        is_public=False,
        tags=[]
    )
    
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    
    return _pipeline_to_response(pipeline)


@router.get("/", response_model=PipelineListResponse)
async def list_pipelines(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    is_public: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    List user's pipelines with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        category: Filter by category
        is_public: Filter by public status
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Paginated list of pipelines
    """
    query = db.query(Pipeline).filter(Pipeline.user_id == current_user.user_id)
    
    if category:
        query = query.filter(Pipeline.category == category)
    if is_public is not None:
        query = query.filter(Pipeline.is_public == is_public)
    
    total = query.count()
    pipelines = query.order_by(Pipeline.last_modified.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    
    return PipelineListResponse(
        pipelines=[_pipeline_to_response(p) for p in pipelines],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/public", response_model=PipelineListResponse)
async def list_public_pipelines(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Any:
    """
    Browse public pipeline templates.
    
    Args:
        page: Page number
        page_size: Items per page
        category: Filter by category
        search: Search in name/description
        db: Database session
        
    Returns:
        Paginated list of public pipelines
    """
    query = db.query(Pipeline).filter(Pipeline.is_public == True)
    
    if category:
        query = query.filter(Pipeline.category == category)
    if search:
        query = query.filter(
            Pipeline.pipeline_name.ilike(f"%{search}%") |
            Pipeline.description.ilike(f"%{search}%")
        )
    
    total = query.count()
    pipelines = query.order_by(Pipeline.usage_count.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    
    return PipelineListResponse(
        pipelines=[_pipeline_to_response(p) for p in pipelines],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get pipeline details.
    
    Args:
        pipeline_id: UUID of the pipeline
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Pipeline details
    """
    pipeline = db.query(Pipeline).filter(Pipeline.pipeline_id == pipeline_id).first()
    
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline not found"
        )
    
    # Check access
    if not pipeline.is_public and pipeline.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return _pipeline_to_response(pipeline)


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: UUID,
    pipeline_data: PipelineUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update a pipeline.
    
    Args:
        pipeline_id: UUID of the pipeline
        pipeline_data: Update data
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated pipeline
    """
    pipeline = db.query(Pipeline).filter(
        Pipeline.pipeline_id == pipeline_id,
        Pipeline.user_id == current_user.user_id
    ).first()
    
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline not found"
        )
    
    # Update fields
    if pipeline_data.pipeline_name is not None:
        pipeline.pipeline_name = pipeline_data.pipeline_name
    if pipeline_data.description is not None:
        pipeline.description = pipeline_data.description
    if pipeline_data.category is not None:
        pipeline.category = pipeline_data.category
    if pipeline_data.transformations is not None:
        pipeline.transformations_json = pipeline_data.transformations
    if pipeline_data.config is not None:
        pipeline.config_json = pipeline_data.config
    if pipeline_data.is_public is not None:
        pipeline.is_public = pipeline_data.is_public
    if pipeline_data.tags is not None:
        pipeline.tags = pipeline_data.tags
    
    db.commit()
    db.refresh(pipeline)
    
    return _pipeline_to_response(pipeline)


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(
    pipeline_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> None:
    """
    Delete a pipeline.
    
    Args:
        pipeline_id: UUID of the pipeline
        current_user: Authenticated user
        db: Database session
    """
    pipeline = db.query(Pipeline).filter(
        Pipeline.pipeline_id == pipeline_id,
        Pipeline.user_id == current_user.user_id
    ).first()
    
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline not found"
        )
    
    db.delete(pipeline)
    db.commit()


@router.post("/{pipeline_id}/apply/{dataset_id}")
async def apply_pipeline_to_dataset(
    pipeline_id: UUID,
    dataset_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Apply a saved pipeline to a new dataset.
    
    Args:
        pipeline_id: UUID of the pipeline to apply
        dataset_id: UUID of the target dataset
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Transformation results
    """
    # Get pipeline
    pipeline = db.query(Pipeline).filter(Pipeline.pipeline_id == pipeline_id).first()
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pipeline not found"
        )
    
    # Check access
    if not pipeline.is_public and pipeline.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get dataset
    dataset = db.query(Dataset).filter(
        Dataset.dataset_id == dataset_id,
        Dataset.user_id == current_user.user_id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Load dataset
    try:
        df = csv_parser.parse_csv(dataset.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading dataset: {str(e)}"
        )
    
    # Apply transformations
    results = []
    for transform in pipeline.transformations_json:
        try:
            result = transformation_service.apply_transformation(
                df=df,
                dataset_id=str(dataset_id),
                transformation_type=transform.get('transformation_type'),
                columns=transform.get('parameters', {}).get('columns', []),
                **{k: v for k, v in transform.get('parameters', {}).items() if k != 'columns'}
            )
            results.append({
                "transformation": transform.get('transformation_type'),
                "success": True,
                "rows_affected": result.rows_affected
            })
            df = result.result_df
        except Exception as e:
            results.append({
                "transformation": transform.get('transformation_type'),
                "success": False,
                "error": str(e)
            })
    
    # Increment usage count
    pipeline.usage_count += 1
    db.commit()
    
    return {
        "pipeline_id": str(pipeline_id),
        "dataset_id": str(dataset_id),
        "transformations_applied": len(results),
        "results": results
    }


def _pipeline_to_response(pipeline: Pipeline) -> PipelineResponse:
    """Convert Pipeline model to response schema."""
    return PipelineResponse(
        pipeline_id=pipeline.pipeline_id,
        user_id=pipeline.user_id,
        dataset_id=pipeline.dataset_id,
        pipeline_name=pipeline.pipeline_name,
        description=pipeline.description,
        category=pipeline.category,
        transformations=pipeline.transformations_json or [],
        config=pipeline.config_json,
        created_at=pipeline.created_at,
        last_modified=pipeline.last_modified,
        is_public=pipeline.is_public,
        usage_count=pipeline.usage_count,
        tags=pipeline.tags or []
    )
