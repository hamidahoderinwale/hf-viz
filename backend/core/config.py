"""Configuration management."""
import os
from typing import Optional

class Settings:
    """Application settings."""
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    ALLOW_ALL_ORIGINS: bool = os.getenv("ALLOW_ALL_ORIGINS", "True").lower() in ("true", "1", "yes")
    SAMPLE_SIZE: Optional[int] = None
    USE_GRAPH_EMBEDDINGS: bool = os.getenv("USE_GRAPH_EMBEDDINGS", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", 8000))
    
    @classmethod
    def get_sample_size(cls) -> Optional[int]:
        """Get sample size from environment."""
        sample_size_env = os.getenv("SAMPLE_SIZE")
        if sample_size_env:
            sample_size_val = int(sample_size_env)
            return sample_size_val if sample_size_val > 0 else None
        return None

settings = Settings()

