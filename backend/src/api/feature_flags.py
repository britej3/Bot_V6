from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.src.models.feature_flag import FeatureFlag
from backend.src.services.feature_flag_service import feature_flag_service

router = APIRouter()

class FeatureFlagUpdate(BaseModel):
    enabled: bool

@router.get("/feature-flags", response_model=List[FeatureFlag])
async def get_feature_flags():
    return feature_flag_service.get_all_flags()

@router.put("/feature-flags/{name}", response_model=FeatureFlag)
async def update_feature_flag(name: str, update_data: FeatureFlagUpdate):
    flag = feature_flag_service.update_flag(name, update_data.enabled)
    if not flag:
        raise HTTPException(status_code=404, detail="Feature flag not found")
    return flag