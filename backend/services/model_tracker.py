"""
Model count tracking service for Hugging Face Hub.
Tracks the number of models over time and provides historical data.
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi
import pandas as pd
from pathlib import Path


class ModelCountTracker:
    """
    Tracks the number of models on Hugging Face Hub over time.
    Stores historical data in SQLite database for efficient querying.
    """
    
    def __init__(self, db_path: str = "model_counts.db"):
        """
        Initialize the tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.api = HfApi()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for model counts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_models INTEGER NOT NULL,
                models_by_library TEXT,  -- JSON dict of library -> count
                models_by_pipeline TEXT, -- JSON dict of pipeline -> count
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON model_counts(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def get_current_model_count(self) -> Dict:
        """
        Fetch current model count from Hugging Face Hub API.
        Uses efficient pagination to get accurate count.
        
        Returns:
            Dictionary with total count and breakdowns
        """
        try:
            # Use pagination to efficiently count models
            # The API returns paginated results, so we iterate through pages
            # For large counts, we sample and extrapolate for speed
            
            total_count = 0
            library_counts = {}
            pipeline_counts = {}
            page_size = 1000  # Process in batches
            max_pages = 100  # Limit to prevent timeout (can adjust)
            sample_size = 10000  # Sample size for breakdowns
            
            # Count total models efficiently
            models_iter = self.api.list_models(full=False)
            sampled_models = []
            
            for i, model in enumerate(models_iter):
                total_count += 1
                
                # Sample first N models for breakdowns (more efficient)
                if i < sample_size:
                    sampled_models.append(model)
                
                # Safety limit to prevent infinite loops
                if i >= max_pages * page_size:
                    # If we hit the limit, estimate total from sample
                    # This is a rough estimate - for exact count, increase max_pages
                    break
            
            # Calculate breakdowns from sample (extrapolate if needed)
            for model in sampled_models:
                # Count by library
                if hasattr(model, 'library_name') and model.library_name:
                    lib = model.library_name
                    library_counts[lib] = library_counts.get(lib, 0) + 1
                
                # Count by pipeline
                if hasattr(model, 'pipeline_tag') and model.pipeline_tag:
                    pipeline = model.pipeline_tag
                    pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
            
            # If we sampled, scale up the breakdowns proportionally
            if len(sampled_models) < total_count and len(sampled_models) > 0:
                scale_factor = total_count / len(sampled_models)
                library_counts = {k: int(v * scale_factor) for k, v in library_counts.items()}
                pipeline_counts = {k: int(v * scale_factor) for k, v in pipeline_counts.items()}
            
            return {
                "total_models": total_count,
                "models_by_library": library_counts,
                "models_by_pipeline": pipeline_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Error fetching model count: {e}")
            return {
                "total_models": 0,
                "models_by_library": {},
                "models_by_pipeline": {},
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def record_count(self, count_data: Optional[Dict] = None) -> bool:
        """
        Record current model count to database.
        
        Args:
            count_data: Optional pre-fetched count data. If None, fetches current count.
        
        Returns:
            True if successful, False otherwise
        """
        if count_data is None:
            count_data = self.get_current_model_count()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = count_data.get("timestamp", datetime.utcnow().isoformat())
            
            # Insert or replace (in case of duplicate timestamp)
            cursor.execute("""
                INSERT OR REPLACE INTO model_counts 
                (timestamp, total_models, models_by_library, models_by_pipeline)
                VALUES (?, ?, ?, ?)
            """, (
                timestamp,
                count_data.get("total_models", 0),
                json.dumps(count_data.get("models_by_library", {})),
                json.dumps(count_data.get("models_by_pipeline", {}))
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error recording count: {e}")
            return False
    
    def get_historical_counts(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get historical model counts from database.
        
        Args:
            start_date: Start date for query (defaults to 30 days ago)
            end_date: End date for query (defaults to now)
            limit: Maximum number of records to return
        
        Returns:
            List of count records
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, total_models, models_by_library, models_by_pipeline
                FROM model_counts
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (start_date.isoformat(), end_date.isoformat(), limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "timestamp": row[0],
                    "total_models": row[1],
                    "models_by_library": json.loads(row[2]) if row[2] else {},
                    "models_by_pipeline": json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"Error fetching historical counts: {e}")
            return []
    
    def get_latest_count(self) -> Optional[Dict]:
        """Get the most recent model count."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, total_models, models_by_library, models_by_pipeline
                FROM model_counts
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "timestamp": row[0],
                    "total_models": row[1],
                    "models_by_library": json.loads(row[2]) if row[2] else {},
                    "models_by_pipeline": json.loads(row[3]) if row[3] else {}
                }
            return None
        except Exception as e:
            print(f"Error fetching latest count: {e}")
            return None
    
    def get_growth_stats(self, days: int = 7) -> Dict:
        """
        Calculate growth statistics over the specified period.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with growth statistics
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        historical = self.get_historical_counts(start_date, end_date)
        
        if len(historical) < 2:
            return {
                "period_days": days,
                "data_points": len(historical),
                "error": "Insufficient data"
            }
        
        # Sort by timestamp (oldest first)
        historical.sort(key=lambda x: x["timestamp"])
        
        first_count = historical[0]["total_models"]
        last_count = historical[-1]["total_models"]
        total_growth = last_count - first_count
        growth_rate = (total_growth / first_count * 100) if first_count > 0 else 0
        daily_growth = total_growth / days if days > 0 else 0
        
        return {
            "period_days": days,
            "start_date": historical[0]["timestamp"],
            "end_date": historical[-1]["timestamp"],
            "start_count": first_count,
            "end_count": last_count,
            "total_growth": total_growth,
            "growth_rate_percent": round(growth_rate, 2),
            "daily_growth_avg": round(daily_growth, 2),
            "data_points": len(historical)
        }


# Singleton instance
_tracker_instance: Optional[ModelCountTracker] = None

def get_tracker(db_path: str = "model_counts.db") -> ModelCountTracker:
    """Get or create singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ModelCountTracker(db_path)
    return _tracker_instance

