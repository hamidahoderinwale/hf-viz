/**
 * Distance heatmap overlay for 3D visualization.
 * Shows distance gradients from selected point.
 */
import React, { useMemo } from 'react';
import { ModelPoint } from '../../types';

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
  opacity = 0.3
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
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width,
        height,
        pointerEvents: 'none',
        zIndex: 1
      }}
    >
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: '4px',
          fontSize: '11px',
          fontFamily: "'Instrument Sans', sans-serif"
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: '4px' }}>Distance Heatmap</div>
        <div style={{ fontSize: '10px', opacity: 0.9 }}>
          Showing distance from: <strong>{selectedModel.model_id}</strong>
        </div>
        <div style={{ fontSize: '10px', opacity: 0.8, marginTop: '4px' }}>
          Range: {distances.minDist.toFixed(2)} - {distances.maxDist.toFixed(2)}
        </div>
      </div>
    </div>
  );
}


