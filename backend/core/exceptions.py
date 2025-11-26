"""Custom exceptions."""
from fastapi import HTTPException

class ModelNotFoundError(HTTPException):
    """Model not found exception."""
    def __init__(self, model_id: str):
        super().__init__(status_code=404, detail=f"Model not found: {model_id}")

class DataNotLoadedError(HTTPException):
    """Data not loaded exception."""
    def __init__(self):
        super().__init__(status_code=503, detail="Data not loaded")

class EmbeddingsNotReadyError(HTTPException):
    """Embeddings not ready exception."""
    def __init__(self):
        super().__init__(status_code=503, detail="Embeddings not ready")

