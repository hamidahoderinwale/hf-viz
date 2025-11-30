import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config/api';
import LoadingProgress from '../components/ui/LoadingProgress';
import './AnalyticsPage.css';

interface TopModel {
  model_id: string;
  downloads?: number;
  likes?: number;
  trending_score?: number;
  created_at?: string;
}

interface Family {
  family: string;
  count: number;
  growth_rate?: number;
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('30d');
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  
  const [topDownloads, setTopDownloads] = useState<TopModel[]>([]);
  const [topLikes, setTopLikes] = useState<TopModel[]>([]);
  const [trending, setTrending] = useState<TopModel[]>([]);
  const [newest, setNewest] = useState<TopModel[]>([]);
  const [largestFamilies, setLargestFamilies] = useState<Family[]>([]);
  const [fastestGrowing, setFastestGrowing] = useState<Family[]>([]);

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      setLoadingProgress(0);
      
      try {
        // Fetch models data and sort by different criteria
        setLoadingProgress(20);
        const response = await fetch(`${API_BASE}/api/models?max_points=10000&format=json`);
        if (!response.ok) throw new Error('Failed to fetch models');
        
        setLoadingProgress(40);
        const data = await response.json();
        const models: TopModel[] = Array.isArray(data) ? data : (data.models || []);
        
        // Sort by downloads
        setLoadingProgress(50);
        const sortedByDownloads = [...models]
          .sort((a, b) => (b.downloads || 0) - (a.downloads || 0))
          .slice(0, 20);
        setTopDownloads(sortedByDownloads);
        
        // Sort by likes
        setLoadingProgress(60);
        const sortedByLikes = [...models]
          .sort((a, b) => (b.likes || 0) - (a.likes || 0))
          .slice(0, 20);
        setTopLikes(sortedByLikes);
        
        // Sort by trending score
        setLoadingProgress(70);
        const sortedByTrending = [...models]
          .filter(m => m.trending_score !== null && m.trending_score !== undefined)
          .sort((a, b) => (b.trending_score || 0) - (a.trending_score || 0))
          .slice(0, 20);
        setTrending(sortedByTrending);
        
        // Sort by created_at (newest)
        setLoadingProgress(80);
        const sortedByNewest = [...models]
          .filter(m => m.created_at)
          .sort((a, b) => {
            const dateA = new Date(a.created_at || 0).getTime();
            const dateB = new Date(b.created_at || 0).getTime();
            return dateB - dateA;
          })
          .slice(0, 20);
        setNewest(sortedByNewest);
        
        // Group by family (using parent_model or model_id prefix)
        setLoadingProgress(90);
        const familyMap = new Map<string, number>();
        models.forEach(model => {
          // Extract family name from model_id (e.g., "meta-llama/Meta-Llama-3" -> "meta-llama")
          const family = model.model_id.split('/')[0];
          familyMap.set(family, (familyMap.get(family) || 0) + 1);
        });
        
        const families: Family[] = Array.from(familyMap.entries())
          .map(([family, count]) => ({ family, count }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 20);
        setLargestFamilies(families);
        setFastestGrowing(families); // TODO: Calculate actual growth rate
        
        setLoadingProgress(100);
        setLoading(false);
      } catch {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  const renderCardContent = (cardType: string) => {
    switch (cardType) {
      case 'downloads':
        return (
          <div className="card-expanded">
            <div className="card-header">
              <h3>Top Downloads ({timeRange})</h3>
              <div className="time-tabs">
                <button
                  className={`tab ${timeRange === '24h' ? 'active' : ''}`}
                  onClick={(e) => { e.stopPropagation(); setTimeRange('24h'); }}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  24h
                </button>
                <button
                  className={`tab ${timeRange === '7d' ? 'active' : ''}`}
                  onClick={(e) => { e.stopPropagation(); setTimeRange('7d'); }}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  7d
                </button>
                <button
                  className={`tab ${timeRange === '30d' ? 'active' : ''}`}
                  onClick={(e) => { e.stopPropagation(); setTimeRange('30d'); }}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  30d
                </button>
              </div>
            </div>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Downloads</th>
                </tr>
              </thead>
              <tbody>
                {topDownloads.length > 0 ? (
                  topDownloads.map((model, idx) => (
                    <tr key={model.model_id}>
                      <td>{idx + 1}</td>
                      <td title={model.model_id}>{model.model_id}</td>
                      <td>{model.downloads?.toLocaleString() || 'N/A'}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      case 'likes':
        return (
          <div className="card-expanded">
            <h3>Top Likes</h3>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Likes</th>
                </tr>
              </thead>
              <tbody>
                {topLikes.length > 0 ? (
                  topLikes.map((model, idx) => (
                    <tr key={model.model_id}>
                      <td>{idx + 1}</td>
                      <td title={model.model_id}>{model.model_id}</td>
                      <td>{model.likes?.toLocaleString() || 'N/A'}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      case 'trending':
        return (
          <div className="card-expanded">
            <h3>Trending Models</h3>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Trending Score</th>
                </tr>
              </thead>
              <tbody>
                {trending.length > 0 ? (
                  trending.map((model, idx) => (
                    <tr key={model.model_id}>
                      <td>{idx + 1}</td>
                      <td title={model.model_id}>{model.model_id}</td>
                      <td>{model.trending_score?.toFixed(2) || 'N/A'}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      case 'newest':
        return (
          <div className="card-expanded">
            <h3>Newest Models</h3>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {newest.length > 0 ? (
                  newest.map((model, idx) => (
                    <tr key={model.model_id}>
                      <td>{idx + 1}</td>
                      <td title={model.model_id}>{model.model_id}</td>
                      <td>{model.created_at ? new Date(model.created_at).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      case 'largest':
        return (
          <div className="card-expanded">
            <h3>Largest Families</h3>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Family</th>
                  <th>Model Count</th>
                </tr>
              </thead>
              <tbody>
                {largestFamilies.length > 0 ? (
                  largestFamilies.map((family, idx) => (
                    <tr key={family.family}>
                      <td>{idx + 1}</td>
                      <td>{family.family}</td>
                      <td>{family.count.toLocaleString()}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      case 'fastest':
        return (
          <div className="card-expanded">
            <h3>Fastest-Growing Families</h3>
            <table className="analytics-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Family</th>
                  <th>Model Count</th>
                </tr>
              </thead>
              <tbody>
                {fastestGrowing.length > 0 ? (
                  fastestGrowing.map((family, idx) => (
                    <tr key={family.family}>
                      <td>{idx + 1}</td>
                      <td>{family.family}</td>
                      <td>{family.count.toLocaleString()}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={3} className="placeholder">Loading...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        );
      default:
        return <div className="card-placeholder">Content coming soon</div>;
    }
  };

  if (loading) {
    return (
      <LoadingProgress 
        message="Loading analytics..." 
        progress={loadingProgress}
        subMessage="Fetching top models and families"
      />
    );
  }

  return (
    <div className="analytics-page">
      <div className="page-header">
        <h1>Analytics</h1>
      </div>

      <div className="analytics-grid">
        <div className="analytics-card expanded">
          {renderCardContent('downloads')}
        </div>

        <div className="analytics-card expanded">
          {renderCardContent('likes')}
        </div>

        <div className="analytics-card expanded">
          {renderCardContent('trending')}
        </div>

        <div className="analytics-card expanded">
          {renderCardContent('newest')}
        </div>

        <div className="analytics-card expanded">
          {renderCardContent('largest')}
        </div>

        <div className="analytics-card expanded">
          {renderCardContent('fastest')}
        </div>
      </div>
    </div>
  );
}
