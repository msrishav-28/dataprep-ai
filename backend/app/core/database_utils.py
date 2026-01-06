"""
Database utility functions for initialization and management.
"""
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.models.user import User
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

DUMMY_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
DUMMY_USER_EMAIL = "demo@dataprep.ai"


def create_dummy_user() -> None:
    """Create a dummy user for MVP testing purposes."""
    db = SessionLocal()
    try:
        # Check if dummy user already exists
        existing_user = db.query(User).filter(User.user_id == DUMMY_USER_ID).first()
        
        if not existing_user:
            dummy_user = User(
                user_id=DUMMY_USER_ID,
                email=DUMMY_USER_EMAIL,
                is_active=True
            )
            db.add(dummy_user)
            db.commit()
            logger.info(f"Created dummy user: {DUMMY_USER_EMAIL}")
        else:
            logger.info("Dummy user already exists")
            
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating dummy user: {e}")
    finally:
        db.close()


def get_dummy_user_id() -> UUID:
    """Get the dummy user ID for MVP purposes."""
    return DUMMY_USER_ID