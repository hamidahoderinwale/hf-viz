"""
Improved model count tracking service for Hugging Face Hub.
Based on investigation of ai-ecosystem repo patterns and Hugging Face API best practices.

Key improvements:
- More efficient API usage with caching
- Better handling of large model counts
- Support for incremental updates
- Integration with dataset snapshots
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
import time
import httpx

logger = logging.getLogger(__name__)


class ImprovedModelCountTracker:
    """
    Improved tracker that uses multiple strategies for efficiency:
    1. Direct API count (when available)
    2. Pagination with sampling for breakdowns
    3. Caching to avoid repeated expensive calls
    4. Integration with dataset snapshots
    """
    
    def __init__(self, db_path: str = "model_counts.db", cache_ttl_seconds: int = 300):
        """
        Initialize the tracker.
        
        Args:
            db_path: Path to SQLite database file
            cache_ttl_seconds: Time-to-live for in-memory cache (default 5 minutes)
        """
        self.db_path = db_path
        self.api = HfApi()
        self.cache_ttl = cache_ttl_seconds
        self._cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
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
                models_by_author TEXT,   -- JSON dict of author -> count (top authors)
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
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache is None or self._cache_timestamp is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return elapsed < self.cache_ttl
    
    def get_current_model_count(self, use_cache: bool = True, force_refresh: bool = False, use_models_page: bool = True) -> Dict:
        """
        Fetch current model count from Hugging Face Hub.
        Uses multiple strategies: models page scraping (fastest), API, or dataset snapshot.
        
        Args:
            use_cache: Whether to use cached results if available
            force_refresh: Force refresh even if cache is valid
            use_models_page: Try to get count from HF models page first (default: True)
        
        Returns:
            Dictionary with total count and breakdowns
        """
        if use_cache and not force_refresh and self._is_cache_valid():
            return self._cache
        
        if use_models_page:
            page_count = self.get_count_from_models_page()
            if page_count:
                dataset_count = self.get_count_from_dataset_snapshot()
                if dataset_count and dataset_count.get("models_by_library"):
                    page_count["models_by_library"] = dataset_count.get("models_by_library", {})
                    page_count["models_by_pipeline"] = dataset_count.get("models_by_pipeline", {})
                    page_count["models_by_author"] = dataset_count.get("models_by_author", {})
                
                self._cache = page_count
                self._cache_timestamp = datetime.utcnow()
                return page_count
        
        try:
            total_count = 0
            library_counts = {}
            pipeline_counts = {}
            author_counts = {}
            
            sample_size = 20000
            max_count_for_full_breakdown = 50000
            
            models_iter = self.api.list_models(full=False, sort="created", direction=-1)
            sampled_models = []
            
            start_time = time.time()
            timeout_seconds = 30
            
            for i, model in enumerate(models_iter):
                if time.time() - start_time > timeout_seconds:
                    break
                
                total_count += 1
                
                if i < sample_size:
                    sampled_models.append(model)
                
                if total_count < max_count_for_full_breakdown:
                    if hasattr(model, 'library_name') and model.library_name:
                        lib = model.library_name
                        library_counts[lib] = library_counts.get(lib, 0) + 1
                    
                    if hasattr(model, 'pipeline_tag') and model.pipeline_tag:
                        pipeline = model.pipeline_tag
                        pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
                    
                    if hasattr(model, 'id') and model.id:
                        author = model.id.split('/')[0] if '/' in model.id else 'unknown'
                        author_counts[author] = author_counts.get(author, 0) + 1
            
            if total_count > len(sampled_models) and len(sampled_models) > 0:
                for model in sampled_models:
                    if hasattr(model, 'library_name') and model.library_name:
                        lib = model.library_name
                        library_counts[lib] = library_counts.get(lib, 0) + 1
                    
                    if hasattr(model, 'pipeline_tag') and model.pipeline_tag:
                        pipeline = model.pipeline_tag
                        pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
                    
                    if hasattr(model, 'id') and model.id:
                        author = model.id.split('/')[0] if '/' in model.id else 'unknown'
                        author_counts[author] = author_counts.get(author, 0) + 1
                
                if len(sampled_models) > 0:
                    scale_factor = total_count / len(sampled_models)
                    library_counts = {k: int(v * scale_factor) for k, v in library_counts.items()}
                    pipeline_counts = {k: int(v * scale_factor) for k, v in pipeline_counts.items()}
                    author_counts = {k: int(v * scale_factor) for k, v in author_counts.items()}
            
            result = {
                "total_models": total_count,
                "models_by_library": library_counts,
                "models_by_pipeline": pipeline_counts,
                "models_by_author": dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
                "timestamp": datetime.utcnow().isoformat(),
                "sampling_used": total_count > len(sampled_models) if sampled_models else False,
                "sample_size": len(sampled_models) if sampled_models else total_count
            }
            
            self._cache = result
            self._cache_timestamp = datetime.utcnow()
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching model count: {e}", exc_info=True)
            return {
                "total_models": 0,
                "models_by_library": {},
                "models_by_pipeline": {},
                "models_by_author": {},
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def get_count_from_models_page(self) -> Optional[Dict]:
        """
        Get model count by scraping the Hugging Face models page.
        Extracts count from the div with class "font-normal text-gray-400" on https://huggingface.co/models
        
        Returns:
            Dictionary with total_models count, or None if extraction fails
        """
        try:
            url = "https://huggingface.co/models"
            response = httpx.get(url, timeout=10.0, follow_redirects=True)
            response.raise_for_status()
            
            html_content = response.text
            
            # Look for the pattern: <div class="font-normal text-gray-400">2,249,310</div>
            # The number is in the format with commas
            pattern = r'<div[^>]*class="[^"]*font-normal[^"]*text-gray-400[^"]*"[^>]*>([\d,]+)</div>'
            matches = re.findall(pattern, html_content)
            
            if matches:
                # Take the first match and remove commas
                count_str = matches[0].replace(',', '')
                total_models = int(count_str)
                
                logger.info(f"Extracted model count from HF models page: {total_models}")
                
                return {
                    "total_models": total_models,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "hf_models_page",
                    "models_by_library": {},
                    "models_by_pipeline": {},
                    "models_by_author": {}
                }
            else:
                # Fallback: try to find the number in the window.__hf_deferred object
                # The page has: window.__hf_deferred["numTotalItems"] = 2249312;
                deferred_pattern = r'window\.__hf_deferred\["numTotalItems"\]\s*=\s*(\d+);'
                deferred_matches = re.findall(deferred_pattern, html_content)
                
                if deferred_matches:
                    total_models = int(deferred_matches[0])
                    logger.info(f"Extracted model count from window.__hf_deferred: {total_models}")
                    
                    return {
                        "total_models": total_models,
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "hf_models_page_deferred",
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
    
    def get_count_from_dataset_snapshot(self, dataset_name: str = "modelbiome/ai_ecosystem_withmodelcards") -> Optional[Dict]:
        """
        Alternative method: Get count from dataset snapshot (like ai-ecosystem repo does).
        This is faster but may be slightly outdated.
        
        Args:
            dataset_name: Name of the Hugging Face dataset to use
        
        Returns:
            Dictionary with count information or None if dataset unavailable
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split="train")
            total_count = len(dataset)
            
            sample_size = min(10000, total_count)
            sample = dataset.shuffle(seed=42).select(range(sample_size))
            
            library_counts = {}
            pipeline_counts = {}
            
            for item in sample:
                if 'library_name' in item and item['library_name']:
                    lib = item['library_name']
                    library_counts[lib] = library_counts.get(lib, 0) + 1
                
                if 'pipeline_tag' in item and item['pipeline_tag']:
                    pipeline = item['pipeline_tag']
                    pipeline_counts[pipeline] = pipeline_counts.get(pipeline, 0) + 1
            
            if sample_size < total_count:
                scale_factor = total_count / sample_size
                library_counts = {k: int(v * scale_factor) for k, v in library_counts.items()}
                pipeline_counts = {k: int(v * scale_factor) for k, v in pipeline_counts.items()}
            
            return {
                "total_models": total_count,
                "models_by_library": library_counts,
                "models_by_pipeline": pipeline_counts,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "dataset_snapshot"
            }
        except Exception as e:
            logger.error(f"Error loading from dataset snapshot: {e}", exc_info=True)
            return None
    
    def record_count(self, count_data: Optional[Dict] = None, source: str = "api") -> bool:
        """
        Record current model count to database.
        
        Args:
            count_data: Optional pre-fetched count data. If None, fetches current count.
            source: Source of the data ("api" or "dataset_snapshot")
        
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
                (timestamp, total_models, models_by_library, models_by_pipeline, models_by_author)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                count_data.get("total_models", 0),
                json.dumps(count_data.get("models_by_library", {})),
                json.dumps(count_data.get("models_by_pipeline", {})),
                json.dumps(count_data.get("models_by_author", {}))
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
                SELECT timestamp, total_models, models_by_library, models_by_pipeline, models_by_author
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
                    "models_by_pipeline": json.loads(row[3]) if row[3] else {},
                    "models_by_author": json.loads(row[4]) if row[4] else {}
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
                SELECT timestamp, total_models, models_by_library, models_by_pipeline, models_by_author
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
                    "models_by_pipeline": json.loads(row[3]) if row[3] else {},
                    "models_by_author": json.loads(row[4]) if row[4] else {}
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
_improved_tracker_instance: Optional[ImprovedModelCountTracker] = None

def get_improved_tracker(db_path: str = "model_counts.db") -> ImprovedModelCountTracker:
    """Get or create singleton improved tracker instance."""
    global _improved_tracker_instance
    if _improved_tracker_instance is None:
        _improved_tracker_instance = ImprovedModelCountTracker(db_path)
    return _improved_tracker_instance

