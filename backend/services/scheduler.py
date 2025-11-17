"""
Background scheduler for periodic tasks like model count tracking.
Can be run as a separate process or integrated into the main API.
"""
import time
import schedule
from datetime import datetime
from services.model_tracker import get_tracker
from services.model_tracker_improved import get_improved_tracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def record_model_count_job(use_dataset_snapshot: bool = False):
    """
    Job to record current model count.
    
    Args:
        use_dataset_snapshot: If True, use dataset snapshot (faster, like ai-ecosystem repo)
    """
    try:
        logger.info(f"Running model count recording job at {datetime.utcnow()}")
        tracker = get_improved_tracker()
        
        if use_dataset_snapshot:
            # Try dataset snapshot first (faster, like ai-ecosystem approach)
            count_data = tracker.get_count_from_dataset_snapshot()
            if count_data is None:
                logger.warning("Dataset snapshot unavailable, falling back to API")
                count_data = tracker.get_current_model_count(use_cache=False)
                source = "api"
            else:
                source = "dataset_snapshot"
        else:
            count_data = tracker.get_current_model_count(use_cache=False)
            source = "api"
        
        if "error" not in count_data:
            success = tracker.record_count(count_data, source=source)
            if success:
                logger.info(f"Recorded model count: {count_data['total_models']} models (source: {source})")
            else:
                logger.error("Failed to record model count")
        else:
            logger.error(f"Error fetching model count: {count_data.get('error')}")
    except Exception as e:
        logger.error(f"Error in model count recording job: {e}")


def setup_scheduler(interval_hours: int = 6, use_dataset_snapshot: bool = False):
    """
    Setup periodic jobs.
    
    Args:
        interval_hours: How often to record model count (in hours)
        use_dataset_snapshot: Use dataset snapshot instead of API (faster, like ai-ecosystem repo)
    """
    # Schedule model count recording
    schedule.every(interval_hours).hours.do(record_model_count_job, use_dataset_snapshot=use_dataset_snapshot)
    
    source = "dataset snapshot" if use_dataset_snapshot else "API"
    logger.info(f"Scheduler configured to record model counts every {interval_hours} hours using {source}")


def run_scheduler(use_dataset_snapshot: bool = False):
    """
    Run the scheduler (blocking).
    
    Args:
        use_dataset_snapshot: Use dataset snapshot instead of API (faster, like ai-ecosystem repo)
    """
    import sys
    
    # Check command line args
    if '--use-dataset-snapshot' in sys.argv:
        use_dataset_snapshot = True
    
    logger.info("Starting scheduler...")
    setup_scheduler(interval_hours=6, use_dataset_snapshot=use_dataset_snapshot)
    
    # Run initial job
    record_model_count_job(use_dataset_snapshot=use_dataset_snapshot)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    run_scheduler()

