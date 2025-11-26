/**
 * Enhanced cluster filter component with search, Select All/Clear All/Random buttons.
 * Inspired by LAION's cluster filtering UI.
 */
import React, { useState, useMemo } from 'react';
import { useFilterStore } from '../../stores/filterStore';
import './ClusterFilter.css';

export interface Cluster {
  cluster_id: number;
  cluster_label: string;
  count: number;
  color?: string;
}

interface ClusterFilterProps {
  clusters: Cluster[];
  loading?: boolean;
}

export default function ClusterFilter({ clusters, loading = false }: ClusterFilterProps) {
  const { selectedClusters, setSelectedClusters } = useFilterStore();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredClusters = useMemo(() => {
    if (!searchTerm) return clusters;
    const lowerSearch = searchTerm.toLowerCase();
    return clusters.filter(c => 
      c.cluster_label.toLowerCase().includes(lowerSearch) ||
      c.cluster_id.toString().includes(lowerSearch)
    );
  }, [clusters, searchTerm]);

  const handleSelectAll = () => {
    setSelectedClusters(clusters.map(c => c.cluster_id));
  };

  const handleClearAll = () => {
    setSelectedClusters([]);
  };

  const handleRandom = () => {
    if (clusters.length === 0) return;
    const randomCluster = clusters[Math.floor(Math.random() * clusters.length)];
    setSelectedClusters([randomCluster.cluster_id]);
  };

  const handleToggleCluster = (clusterId: number) => {
    if (selectedClusters.includes(clusterId)) {
      setSelectedClusters(selectedClusters.filter(id => id !== clusterId));
    } else {
      setSelectedClusters([...selectedClusters, clusterId]);
    }
  };

  if (loading) {
    return (
      <div className="cluster-filter">
        <div className="cluster-filter-loading">Loading clusters...</div>
      </div>
    );
  }

  if (clusters.length === 0) {
    return (
      <div className="cluster-filter">
        <div className="cluster-filter-empty">No clusters available</div>
      </div>
    );
  }

  return (
    <div className="cluster-filter">
      <div className="cluster-filter-header">
        <h3>Dataset Clusters</h3>
      </div>
      
      <div className="cluster-filter-search">
        <input
          type="text"
          placeholder={`Search ${clusters.length} clusters...`}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="cluster-search-input"
        />
      </div>

      <div className="cluster-filter-actions">
        <button
          onClick={handleSelectAll}
          className="cluster-action-btn"
          disabled={clusters.length === 0}
        >
          Select All
        </button>
        <button
          onClick={handleClearAll}
          className="cluster-action-btn"
          disabled={selectedClusters.length === 0}
        >
          Clear All
        </button>
        <button
          onClick={handleRandom}
          className="cluster-action-btn"
          disabled={clusters.length === 0}
        >
          Random
        </button>
      </div>

      <div className="cluster-list">
        {filteredClusters.length === 0 ? (
          <div className="cluster-filter-empty">No clusters match your search</div>
        ) : (
          filteredClusters.map(cluster => (
            <label
              key={cluster.cluster_id}
              className={`cluster-item ${selectedClusters.includes(cluster.cluster_id) ? 'selected' : ''}`}
            >
              <input
                type="checkbox"
                checked={selectedClusters.includes(cluster.cluster_id)}
                onChange={() => handleToggleCluster(cluster.cluster_id)}
                className="cluster-checkbox"
              />
              {cluster.color && (
                <span
                  className="cluster-color-indicator"
                  style={{ backgroundColor: cluster.color }}
                />
              )}
              <span className="cluster-label">{cluster.cluster_label}</span>
              <span className="cluster-count">({cluster.count.toLocaleString()})</span>
            </label>
          ))
        )}
      </div>
    </div>
  );
}

