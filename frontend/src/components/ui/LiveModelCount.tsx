/**
 * Compact live model count display for header.
 * Shows current model count from Hugging Face Hub with auto-refresh.
 * Gets live count from HF models page, breakdowns from dataset.
 */
import React, { useState, useEffect } from 'react';
import { API_BASE } from '../../config/api';
import './LiveModelCount.css';

const logger = {
  error: (message: string, error?: unknown) => {
    if (process.env.NODE_ENV === 'development') {
      console.error(message, error);
    }
  },
};

interface ModelCount {
  total_models: number;
  timestamp: string;
  source?: string;
}

export default function LiveModelCount({ compact = true }: { compact?: boolean }) {
  const [currentCount, setCurrentCount] = useState<ModelCount | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchCurrentCount = async () => {
    try {
      // First try: Get count from HF models page (fastest, most accurate)
      // This gets the live count from https://huggingface.co/models
      // Breakdowns come from dataset snapshot to maintain database structure
      const response = await fetch(`${API_BASE}/api/model-count/current?use_models_page=true&use_dataset_snapshot=true&use_cache=true`);
      if (!response.ok) throw new Error('Failed to fetch model count');
      const data = await response.json();
      setCurrentCount(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      // Fallback: Try without models page scraping
      try {
        const response = await fetch(`${API_BASE}/api/model-count/current?use_cache=true`);
        if (response.ok) {
          const data = await response.json();
          setCurrentCount(data);
          setLastUpdate(new Date());
          setError(null);
        }
      } catch (fallbackErr) {
        logger.error('Fallback fetch also failed:', fallbackErr);
      }
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchCurrentCount();
    setLoading(false);

    // Auto-refresh every 5 minutes
    const interval = setInterval(() => {
      fetchCurrentCount();
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const getTimeAgo = (date: Date) => {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  if (compact) {
    return (
      <div className="live-model-count-compact">
        {loading && !currentCount ? (
          <div className="count-loading">Loading...</div>
        ) : error && !currentCount ? (
          <div className="count-error" title={error}>Error</div>
        ) : currentCount ? (
          <div className="count-badge">
            <span className="count-label">Live Models:</span>
            <span className="count-value">{formatNumber(currentCount.total_models)}</span>
            {lastUpdate && (
              <span className="count-update" title={new Date(currentCount.timestamp).toLocaleString()}>
                {getTimeAgo(lastUpdate)}
              </span>
            )}
          </div>
        ) : null}
      </div>
    );
  }

  // Full version (for sidebar or detailed view)
  return (
    <div className="live-model-count-full">
      <div className="count-header">
        <h4>Live Model Count</h4>
      </div>
      {loading && !currentCount ? (
        <div className="count-loading-full">Loading...</div>
      ) : error && !currentCount ? (
        <div className="count-error-full">
          <div className="count-error-message">Error: {error}</div>
          <button className="count-retry-btn" onClick={fetchCurrentCount}>
            Retry
          </button>
        </div>
      ) : currentCount ? (
        <div className="count-content">
          <div className="count-display">
            <div className="count-number">{formatNumber(currentCount.total_models)}</div>
            <div className="count-label-full">Models on Hugging Face</div>
            {lastUpdate && (
              <div className="count-timestamp">
                Updated {getTimeAgo(lastUpdate)}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

