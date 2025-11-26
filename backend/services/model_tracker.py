"""
Model count tracking service for Hugging Face Hub.
Tracks the number of models over time and provides historical data.
"""
import os
import json
import sqlite3
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi
import pandas as pd
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)


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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON model_counts(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def get_count_from_models_page(self) -> Optional[Dict]:
        """
        Get model count by scraping the Hugging Face models page.
        Extracts count from the div with class "font-normal text-gray-400" on https://huggingface.co/models
        or from window.__hf_deferred["numTotalItems"] in the page script.
        
        Returns:
            Dictionary with total_models count, or None if extraction fails
        """
        try:
            url = "https://huggingface.co/models"
            response = httpx.get(url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            
            html_content = response.text
            
            deferred_pattern = r'window\.__hf_deferred\["numTotalItems"\]\s*=\s*(\d+);'
            deferred_matches = re.findall(deferred_pattern, html_content)
            
            if deferred_matches:
                total_models = int(deferred_matches[0])
                logger.info(f"Extracted model count from window.__hf_deferred: {total_models}")
                
                return {
                    "total_models": total_models,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "hf_models_page",
                    "models_by_library": {},
                    "models_by_pipeline": {},
                    "models_by_author": {}
                }
            
            pattern = r'<div[^>]*class="[^"]*font-normal[^"]*text-gray-400[^"]*"[^>]*>([\d,]+)</div>'
            matches = re.findall(pattern, html_content)
            
            if matches:
                count_str = matches[0].replace(',', '')
                total_models = int(count_str)
                
                logger.info(f"Extracted model count from div: {total_models}")
                
                return {
                    "total_models": total_models,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "hf_models_page",
                    "models_by_library": {},
                    "models_by_pipeline": {},
                    "models_by_author": {}
                }
            
            logger.warning("Could not find model count in HF models page HTML")
            return None
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching HF models page: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error extracting count from HF models page: {e}", exc_info=True)
            return None
    
    def get_current_model_count(self, use_models_page: bool = True) -> Dict:
        """
        Fetch current model count from Hugging Face Hub.
        Uses multiple strategies: models page scraping (fastest), then API enumeration.
        
        Args:
            use_models_page: Try to get count from HF models page first (default: True)
        
        Returns:
            Dictionary with total count and breakdowns
        """
        if use_models_page:
            page_count = self.get_count_from_models_page()
            if page_count:
                return page_count
        
        try:
            total_count = 0
            library_counts = {}
            pipeline_counts = {}
            page_size = 1000
            max_pages = 100
            sample_size = 10000
            
            models_iter = self.api.list_models(full=False)
            sampled_models = []
            
            for i, model in enumerate(models_iter):
                total_count += 1
                
                # Sample first N models for breakdowns (more efficient)
                if i < sample_size:
                    sampled_models.append(model)
                
                if i >= max_pages * page_size:
                    break
            
            for model in sampled_models:
                if hasattr(model, 'library_name') and model.library_name:
                    lib = model.library_name
                    library_counts[lib] = library_counts.get(lib, 0) + 1
                
                if hasattr(model, 'pipeline_tag') and model.pipeline_tag:
                    pipeline = model.pipeline_tag
                    pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
            
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
            logger.error(f"Error fetching model count: {e}", exc_info=True)
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
            logger.error(f"Error recording count: {e}", exc_info=True)
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
            logger.error(f"Error fetching historical counts: {e}", exc_info=True)
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
            logger.error(f"Error fetching latest count: {e}", exc_info=True)
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

