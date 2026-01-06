"""
Authentication dependencies for FastAPI.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token, TokenData
from app.models.user import User

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        User object if authenticated
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_access_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.user_id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional authentication dependency - returns None if not authenticated.
    Useful for endpoints that work for both anonymous and authenticated users.
    
    Args:
        token: Optional JWT token
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
    """
    if token is None:
        return None
    
    token_data = decode_access_token(token)
    if token_data is None:
        return None
    
    user = db.query(User).filter(User.user_id == token_data.user_id).first()
    if user is None or not user.is_active:
        return None
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure user is active.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        User if active
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def require_subscription_tier(required_tier: str):
    """
    Dependency factory to check subscription tier.
    
    Args:
        required_tier: Required subscription tier (free, pro, team, enterprise)
        
    Returns:
        Dependency function that validates subscription
    """
    tier_levels = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}
    
    async def check_subscription(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        user_level = tier_levels.get(current_user.subscription_tier, 0)
        required_level = tier_levels.get(required_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_tier} subscription or higher"
            )
        return current_user
    
    return check_subscription
