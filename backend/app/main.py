"""
FastAPI main application module.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database_utils import create_dummy_user
from app.schemas.common import HealthCheck

# Initialize FastAPI app
app = FastAPI(
    title="DataPrep AI Platform",
    description="Intelligent data preprocessing and exploratory data analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Create dummy user for MVP
    create_dummy_user()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "DataPrep AI Platform API", "version": "1.0.0"}


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        service="dataprep-ai-backend",
        version="1.0.0"
    )


# Include API routers
from app.api.v1.api import api_router
app.include_router(api_router, prefix="/api/v1")