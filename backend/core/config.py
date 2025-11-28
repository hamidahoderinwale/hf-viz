"""Configuration management."""
import os
from typing import Optional

class Settings:
    """Application settings."""
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    ALLOW_ALL_ORIGINS: bool = os.getenv("ALLOW_ALL_ORIGINS", "True").lower() in ("true", "1", "yes")
    USE_GRAPH_EMBEDDINGS: bool = os.getenv("USE_GRAPH_EMBEDDINGS", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Redis caching
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", 300))  # 5 minutes default
    
    # Sample size - read immediately from environment
    @property
    def SAMPLE_SIZE(self) -> Optional[int]:
        """Get sample size from environment (dynamic property)."""
        sample_size_env = os.getenv("SAMPLE_SIZE")
        if sample_size_env:
            try:
                sample_size_val = int(sample_size_env)
                return sample_size_val if sample_size_val > 0 else None
            except ValueError:
                return None
        return None
    
    @classmethod
    def get_sample_size(cls) -> Optional[int]:
        """Get sample size from environment."""
        sample_size_env = os.getenv("SAMPLE_SIZE")
        if sample_size_env:
            try:
                sample_size_val = int(sample_size_env)
                return sample_size_val if sample_size_val > 0 else None
            except ValueError:
                return None
        return None

settings = Settings()

