/**
 * Interactive color legend component for visualizations.
 * Shows color mappings for categorical and continuous data.
 */
import React from 'react';
import { getCategoricalColorMap, getContinuousColorScale } from '../utils/colors';
import './ColorLegend.css';

interface ColorLegendProps {
  colorBy: string;
  data: any[];
  width?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

export default function ColorLegend({ colorBy, data, width = 200, position = 'top-right' }: ColorLegendProps) {
  if (!data || data.length === 0) return null;

  const isCategorical = colorBy === 'library_name' || colorBy === 'pipeline_tag' || colorBy === 'cluster_id';
  
  // Get unique categories or value range
  let legendItems: Array<{ label: string; color: string }> = [];
  
  if (isCategorical) {
    const categories = Array.from(new Set(data.map((d: any) => {
      if (colorBy === 'library_name') return d.library_name || 'unknown';
      if (colorBy === 'pipeline_tag') return d.pipeline_tag || 'unknown';
      if (colorBy === 'cluster_id') return d.cluster_id !== null ? `Cluster ${d.cluster_id}` : 'No cluster';
      return 'unknown';
    }))).sort();
    
    const colorScheme = colorBy === 'library_name' ? 'library' : colorBy === 'pipeline_tag' ? 'pipeline' : 'default';
    const colorMap = getCategoricalColorMap(categories, colorScheme);
    
    legendItems = categories.map(cat => ({
      label: cat,
      color: colorMap.get(cat) || '#808080'
    }));
  } else if (colorBy === 'family_depth') {
    const depths = data.map((d: any) => d.family_depth ?? 0);
    const maxDepth = Math.max(...depths, 1);
    const scale = getContinuousColorScale(0, maxDepth, 'plasma');
    
    // Create gradient legend
    const steps = 10;
    for (let i = 0; i <= steps; i++) {
      const depth = (i / steps) * maxDepth;
      legendItems.push({
        label: `Depth ${Math.round(depth)}`,
        color: scale(depth)
      });
    }
  } else {
    // Continuous scale (downloads, likes)
    const values = data.map((d: any) => {
      if (colorBy === 'downloads') return d.downloads;
      return d.likes;
    });
    const min = Math.min(...values);
    const max = Math.max(...values);
    const scale = getContinuousColorScale(min, max, 'viridis');
    
    // Create gradient legend
    const steps = 10;
    for (let i = 0; i <= steps; i++) {
      const value = min + (i / steps) * (max - min);
      legendItems.push({
        label: value >= 1000 ? `${(value / 1000).toFixed(1)}K` : Math.round(value).toString(),
        color: scale(value)
      });
    }
  }

  const positionClass = `legend-${position}`;

  return (
    <div className={`color-legend ${positionClass}`} style={{ width }}>
      <div className="legend-header">
        <strong>{colorBy.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong>
      </div>
      <div className="legend-content">
        {isCategorical || colorBy === 'family_depth' ? (
          <div className="legend-categorical">
            {legendItems.map((item, idx) => (
              <div key={idx} className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="legend-label">{item.label}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="legend-continuous">
            <div className="legend-gradient">
              {legendItems.map((item, idx) => (
                <div
                  key={idx}
                  className="legend-gradient-segment"
                  style={{ backgroundColor: item.color }}
                />
              ))}
            </div>
            <div className="legend-labels">
              <span>{legendItems[0].label}</span>
              <span>{legendItems[legendItems.length - 1].label}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

