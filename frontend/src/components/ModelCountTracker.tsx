/**
 * Component to display current model count and growth statistics.
 * Can be integrated into the main App to show live ecosystem stats.
 */
import React, { useState, useEffect } from 'react';
import './ModelCountTracker.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ModelCount {
  total_models: number;
  models_by_library?: Record<string, number>;
  models_by_pipeline?: Record<string, number>;
  timestamp: string;
}

interface GrowthStats {
  period_days: number;
  start_count: number;
  end_count: number;
  total_growth: number;
  growth_rate_percent: number;
  daily_growth_avg: number;
}

export default function ModelCountTracker() {
  const [currentCount, setCurrentCount] = useState<ModelCount | null>(null);
  const [growthStats, setGrowthStats] = useState<GrowthStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchCurrentCount = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/model-count/current`);
      if (!response.ok) throw new Error('Failed to fetch model count');
      const data = await response.json();
      setCurrentCount(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const fetchGrowthStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/model-count/growth?days=7`);
      if (!response.ok) throw new Error('Failed to fetch growth stats');
      const data = await response.json();
      setGrowthStats(data);
    } catch (err) {
      console.error('Error fetching growth stats:', err);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchCurrentCount(), fetchGrowthStats()]);
      setLoading(false);
    };

    loadData();

    // Auto-refresh every 5 minutes if enabled
    let interval: NodeJS.Timeout | null = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchCurrentCount();
        fetchGrowthStats();
      }, 5 * 60 * 1000); // 5 minutes
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  if (loading && !currentCount) {
    return (
      <div className="model-count-tracker">
        <div className="tracker-loading">Loading model count...</div>
      </div>
    );
  }

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="model-count-tracker">
      <div className="tracker-header">
        <h3>Hugging Face Model Ecosystem</h3>
        <label className="auto-refresh-toggle">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto-refresh
        </label>
      </div>

      {error && (
        <div className="tracker-error">
          Error: {error}
          <button onClick={fetchCurrentCount}>Retry</button>
        </div>
      )}

      {currentCount && (
        <div className="tracker-content">
          <div className="current-count">
            <div className="count-value">
              {formatNumber(currentCount.total_models)}
            </div>
            <div className="count-label">Total Models</div>
            <div className="count-timestamp">
              Updated: {formatDate(currentCount.timestamp)}
            </div>
          </div>

          {growthStats && (
            <div className="growth-stats">
              <h4>Growth (Last {growthStats.period_days} Days)</h4>
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-value">
                    +{formatNumber(growthStats.total_growth)}
                  </div>
                  <div className="stat-label">Total Growth</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">
                    {growthStats.growth_rate_percent > 0 ? '+' : ''}
                    {growthStats.growth_rate_percent.toFixed(2)}%
                  </div>
                  <div className="stat-label">Growth Rate</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">
                    +{formatNumber(Math.round(growthStats.daily_growth_avg))}
                  </div>
                  <div className="stat-label">Avg. Daily Growth</div>
                </div>
              </div>
            </div>
          )}

          {currentCount.models_by_library && Object.keys(currentCount.models_by_library).length > 0 && (
            <div className="breakdown">
              <h4>Top Libraries</h4>
              <div className="breakdown-list">
                {Object.entries(currentCount.models_by_library)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([library, count]) => (
                    <div key={library} className="breakdown-item">
                      <span className="breakdown-label">{library}</span>
                      <span className="breakdown-value">{formatNumber(count)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          <button 
            className="refresh-button"
            onClick={() => {
              setLoading(true);
              Promise.all([fetchCurrentCount(), fetchGrowthStats()]).then(() => {
                setLoading(false);
              });
            }}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh Now'}
          </button>
        </div>
      )}
    </div>
  );
}

