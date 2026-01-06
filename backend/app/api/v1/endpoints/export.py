"""
Export API endpoints for code, data, and report downloads.
"""
from typing import Dict, Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse
from sqlalchemy.orm import Session
import io
import json

from app.core.database import get_db
from app.services.code_generator import code_generator
from app.services.report_generator import report_generator
from app.services.transformation_service import transformation_service
from app.services.csv_parser import csv_parser
from app.services.quality_service import quality_service
from app.models.dataset import Dataset

router = APIRouter()


@router.get("/code/{dataset_id}")
def export_python_code(
    dataset_id: UUID,
    style: str = Query("pandas", description="Code style: 'pandas' or 'sklearn'"),
    include_comments: bool = Query(True, description="Include explanatory comments"),
    db: Session = Depends(get_db)
) -> PlainTextResponse:
    """
    Export Python code that reproduces the preprocessing transformations.
    
    Args:
        dataset_id: UUID of the dataset
        style: Code style - "pandas" for direct pandas code, "sklearn" for Pipeline
        include_comments: Whether to include explanatory comments
        db: Database session
        
    Returns:
        Python code as plain text
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get transformation history
    transformations = transformation_service.get_transformation_history(str(dataset_id))
    
    # Generate code
    code = code_generator.generate_python_code(
        transformations=transformations,
        dataset_name=dataset.filename,
        include_comments=include_comments,
        style=style
    )
    
    return PlainTextResponse(
        content=code,
        media_type="text/x-python",
        headers={
            "Content-Disposition": f'attachment; filename="preprocessing_{dataset.filename}.py"'
        }
    )


@router.get("/notebook/{dataset_id}")
def export_jupyter_notebook(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Export a Jupyter notebook that reproduces the preprocessing transformations.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Jupyter notebook in JSON format
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get transformation history
    transformations = transformation_service.get_transformation_history(str(dataset_id))
    
    # Generate notebook
    notebook = code_generator.generate_notebook(
        transformations=transformations,
        dataset_name=dataset.filename
    )
    
    return notebook


@router.get("/data/{dataset_id}")
def export_cleaned_data(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Export the cleaned/transformed dataset as CSV.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        CSV file download
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
    
    # Create CSV buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="cleaned_{dataset.filename}"'
        }
    )


@router.get("/report/{dataset_id}")
def export_html_report(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> HTMLResponse:
    """
    Export a comprehensive HTML report for the dataset.
    
    The report includes:
    - Dataset overview and statistics
    - Quality assessment with issues and recommendations
    - Column-level statistics
    - Transformation history
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Self-contained HTML report
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
    
    # Get quality assessment
    quality_assessment = quality_service.assess_quality(df)
    
    # Get transformation history
    transformations = transformation_service.get_transformation_history(str(dataset_id))
    
    # Generate report
    html = report_generator.generate_html_report(
        df=df,
        profile_data=None,  # Could be fetched from cache
        quality_assessment=quality_assessment,
        transformations=transformations,
        visualizations=None,  # Could be fetched from cache
        dataset_name=dataset.filename
    )
    
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="report_{dataset.filename}.html"'
        }
    )


@router.get("/summary/{dataset_id}")
def get_export_options(
    dataset_id: UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get available export options for a dataset.
    
    Args:
        dataset_id: UUID of the dataset
        db: Database session
        
    Returns:
        Available export options and their descriptions
    """
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get transformation count
    transformations = transformation_service.get_transformation_history(str(dataset_id))
    
    return {
        "dataset_id": str(dataset_id),
        "dataset_name": dataset.filename,
        "transformation_count": len(transformations),
        "export_options": [
            {
                "type": "code",
                "description": "Python script with pandas code",
                "endpoint": f"/api/v1/export/code/{dataset_id}",
                "formats": ["pandas", "sklearn"]
            },
            {
                "type": "notebook",
                "description": "Jupyter notebook (.ipynb)",
                "endpoint": f"/api/v1/export/notebook/{dataset_id}",
                "formats": ["ipynb"]
            },
            {
                "type": "data",
                "description": "Cleaned dataset (CSV)",
                "endpoint": f"/api/v1/export/data/{dataset_id}",
                "formats": ["csv"]
            },
            {
                "type": "report",
                "description": "Comprehensive HTML report",
                "endpoint": f"/api/v1/export/report/{dataset_id}",
                "formats": ["html"]
            }
        ]
    }
