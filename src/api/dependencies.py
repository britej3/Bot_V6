from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer
from typing import Dict, Any
from src.config import get_settings as get_app_settings

security = HTTPBearer()

async def get_api_key(x_api_key: str = Header(...)):
    """Dependency to validate API key from request header."""
    # In a real application, this would validate against a secure store of API keys
    if x_api_key != "YOUR_SUPER_SECRET_API_KEY": # Placeholder for actual API key validation
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return x_api_key

async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current user from token."""
    # In a real application, this would decode and validate the JWT token
    # For now, return a mock user
    return {
        'id': 'test_user',
        'username': 'test_user',
        'roles': ['trader'],
        'authenticated': True
    }

def get_settings():
    """Dependency to get application settings."""
    return get_app_settings()
