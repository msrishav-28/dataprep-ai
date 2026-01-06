"""
Main API router for v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import datasets, analysis, transform, export, auth, pipelines

api_router = APIRouter()

# Include auth endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Include dataset endpoints
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])

# Include analysis endpoints
api_router.include_router(analysis.router, prefix="/analyze", tags=["analysis"])

# Include transformation endpoints
api_router.include_router(transform.router, prefix="/transform", tags=["transform"])

# Include export endpoints
api_router.include_router(export.router, prefix="/export", tags=["export"])

# Include pipeline endpoints
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])


