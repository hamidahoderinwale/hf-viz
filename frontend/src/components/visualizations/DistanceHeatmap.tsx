/**
 * Distance heatmap overlay for 3D visualization.
 * Shows distance gradients from selected point.
 */
import React, { useMemo } from 'react';
import { ModelPoint } from '../../types';
import './DistanceHeatmap.css';

interface DistanceHeatmapProps {
  data: ModelPoint[];
  selectedModel: ModelPoint | null;
  width: number;
  height: number;
  opacity?: number;
}

export default function DistanceHeatmap({
  data,
  selectedModel,
  width,
  height,
}: DistanceHeatmapProps) {
  const distances = useMemo(() => {
    if (!selectedModel) return null;
    
    const dists = data.map(point => {
      const dx = point.x - selectedModel.x;
      const dy = point.y - selectedModel.y;
      const dz = point.z - selectedModel.z;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    });
    
    const maxDist = Math.max(...dists);
    const minDist = Math.min(...dists);
    
    return {
      distances: dists,
      maxDist,
      minDist,
      range: maxDist - minDist || 1
    };
  }, [data, selectedModel]);
  
  if (!selectedModel || !distances) return null;
  
  return (
    <div
      className="distance-heatmap-overlay"
      style={{ width, height }}
    >
      <div className="distance-heatmap-info">
        <div className="distance-heatmap-title">Distance Heatmap</div>
        <div className="distance-heatmap-detail">
          Showing distance from: <strong>{selectedModel.model_id}</strong>
        </div>
        <div className="distance-heatmap-range">
          Range: {distances.minDist.toFixed(2)} - {distances.maxDist.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
