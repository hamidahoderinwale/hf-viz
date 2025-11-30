/**
 * Live model counter component for bottom-left display.
 * Shows real-time model count from Hugging Face Hub with pulse animation.
 * Supports continual updates via polling.
 */
import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, RefreshCw } from 'lucide-react';
import { API_BASE } from '../../config/api';
import './LiveModelCounter.css';

interface ModelCountData {
  total_models: number;
  timestamp: string;
  source?: string;
  models_by_library?: Record<string, number>;
  models_by_pipeline?: Record<string, number>;
}

interface GrowthStats {
  daily_growth_avg?: number;
  growth_rate_percent?: number;
  total_growth?: number;
  period_days?: number;
}

interface LiveModelCounterProps {
  pollInterval?: number; // in milliseconds, default 60 seconds
  showGrowth?: boolean;
  onNewModelsDetected?: (count: number, previousCount: number) => void;
}

export default function LiveModelCounter({
  pollInterval = 60000,
  showGrowth = true,
  onNewModelsDetected,
}: LiveModelCounterProps) {
  const [currentCount, setCurrentCount] = useState<ModelCountData | null>(null);
  const [previousCount, setPreviousCount] = useState<number | null>(null);
  const [growthStats, setGrowthStats] = useState<GrowthStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [newModelsAdded, setNewModelsAdded] = useState<number>(0);
  const [showPulse, setShowPulse] = useState(false);

  const fetchCurrentCount = useCallback(async (isManual = false) => {
    if (isManual) setIsRefreshing(true);
    
    try {
      // Try primary endpoint first
      let data: ModelCountData | null = null;
      
      try {
        const response = await fetch(
          `${API_BASE}/api/model-count/current?use_models_page=true&use_cache=${!isManual}`
        );
        if (response.ok) {
          data = await response.json();
        }
      } catch {
        // Primary endpoint failed, will try fallback
      }
      
      // Fallback to stats endpoint
      if (!data || !data.total_models) {
        const statsResponse = await fetch(`${API_BASE}/api/stats`);
        if (statsResponse.ok) {
          const statsData = await statsResponse.json();
          if (statsData.total_models) {
            data = {
              total_models: statsData.total_models,
              timestamp: new Date().toISOString(),
              source: 'stats'
            };
          }
        }
      }
      
      if (!data || !data.total_models) {
        throw new Error('No model count available');
      }
      
      // Check if new models were added
      if (currentCount && data.total_models > currentCount.total_models) {
        const newCount = data.total_models - currentCount.total_models;
        setNewModelsAdded(prev => prev + newCount);
        setShowPulse(true);
        setTimeout(() => setShowPulse(false), 2000);
        
        if (onNewModelsDetected) {
          onNewModelsDetected(data.total_models, currentCount.total_models);
        }
      }
      
      setPreviousCount(currentCount?.total_models || null);
      setCurrentCount(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, [currentCount, onNewModelsDetected]);

  const fetchGrowthStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/model-count/growth?days=7`);
      if (response.ok) {
        const data = await response.json();
        setGrowthStats(data);
      }
    } catch {
      // Silently fail - growth stats are optional
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchCurrentCount();
    if (showGrowth) {
      fetchGrowthStats();
    }
  }, []);

  // Polling interval
  useEffect(() => {
    const interval = setInterval(() => {
      fetchCurrentCount();
    }, pollInterval);

    return () => clearInterval(interval);
  }, [pollInterval, fetchCurrentCount]);

  // Refresh growth stats less frequently
  useEffect(() => {
    if (!showGrowth) return;
    
    const growthInterval = setInterval(() => {
      fetchGrowthStats();
    }, 5 * 60 * 1000); // Every 5 minutes

    return () => clearInterval(growthInterval);
  }, [showGrowth, fetchGrowthStats]);

  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    }
    return new Intl.NumberFormat('en-US').format(num);
  };

  const formatLargeNumber = (num: number): string => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const getTimeAgo = (date: Date): string => {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  if (loading && !currentCount) {
    return (
      <div className="live-model-counter">
        <div className="counter-loading">
          <RefreshCw size={14} className="spin" />
          <span>Loading...</span>
        </div>
      </div>
    );
  }

  if (error && !currentCount) {
    return (
      <div className="live-model-counter counter-error">
        <span>Error loading count</span>
      </div>
    );
  }

  return (
    <div 
      className={`live-model-counter ${showPulse ? 'pulse' : ''}`}
      title="Total number of public models available on the Hugging Face Hub. Updates periodically."
    >
      <div className="counter-main">
        <div className="counter-content">
          <div className="counter-value" title={currentCount ? `Exact count: ${currentCount.total_models.toLocaleString()} models` : undefined}>
            {currentCount ? formatLargeNumber(currentCount.total_models) : '—'}
          </div>
          <div className="counter-label">
            Models on Hugging Face
          </div>
        </div>
      </div>

      {showGrowth && growthStats && growthStats.daily_growth_avg && (
        <div className="counter-growth">
          <TrendingUp size={12} />
          <span>+{Math.round(growthStats.daily_growth_avg).toLocaleString()}/day</span>
        </div>
      )}

      {newModelsAdded > 0 && (
        <div className="counter-new-models">
          +{newModelsAdded.toLocaleString()} new this session
        </div>
      )}

      {lastUpdate && (
        <div className="counter-timestamp">
          Updated {getTimeAgo(lastUpdate)}
        </div>
      )}
    </div>
  );
}

