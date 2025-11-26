"""Pydantic models for API."""
from pydantic import BaseModel
from typing import Optional

class ModelPoint(BaseModel):
    """Model point in 3D space."""
    model_id: str
    x: float
    y: float
    z: float
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    downloads: int
    likes: int
    trending_score: Optional[float]
    tags: Optional[str]
    parent_model: Optional[str] = None
    licenses: Optional[str] = None
    family_depth: Optional[int] = None
    cluster_id: Optional[int] = None
    created_at: Optional[str] = None  # ISO format date string

