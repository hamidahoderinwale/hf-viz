/**
 * Interactive color legend component for visualizations.
 * Shows color mappings for categorical and continuous data.
 */
import React, { useMemo } from 'react';
import { getCategoricalColorMap, getContinuousColorScale, getDepthColorScale } from '../../utils/rendering/colors';
import './ColorLegend.css';

interface ColorLegendProps {
  colorBy: string;
  data: any[];
  width?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  isDarkMode?: boolean;
}

function ColorLegend({ colorBy, data, width = 200, position = 'top-right', isDarkMode }: ColorLegendProps) {
  // Memoize legend calculation to prevent recalculation on every render
  const { legendItems, isContinuous, isCategorical, positionClass } = useMemo(() => {
    if (!data || data.length === 0) {
      return { legendItems: [], isContinuous: false, isCategorical: false, positionClass: `legend-${position}` };
    }

    let isCat = colorBy === 'library_name' || colorBy === 'pipeline_tag' || colorBy === 'cluster_id';
    let items: Array<{ label: string; color: string }> = [];
    let isCont = false;
    
    if (isCat) {
      const categories = Array.from(new Set(data.map((d: any) => {
        if (colorBy === 'library_name') return d.library_name || 'unknown';
        if (colorBy === 'pipeline_tag') return d.pipeline_tag || 'unknown';
        if (colorBy === 'cluster_id') return d.cluster_id !== null ? `Cluster ${d.cluster_id}` : 'No cluster';
        return 'unknown';
      }))).sort();
      
      const colorSchemeType = colorBy === 'library_name' ? 'library' : colorBy === 'pipeline_tag' ? 'pipeline' : 'default';
      const colorMap = getCategoricalColorMap(categories, colorSchemeType);
      
      // Limit to top 20 categories for readability
      const topCategories = categories.slice(0, 20);
      items = topCategories.map(cat => ({
        label: cat.length > 25 ? cat.substring(0, 22) + '...' : cat,
        color: colorMap.get(cat) || '#60a5fa'
      }));
    } else if (colorBy === 'family_depth') {
      isCont = true;
      const depths = data.map((d: any) => d.family_depth ?? 0);
      const maxDepth = Math.max(...depths, 1);
      const minDepth = Math.min(...depths);
      const uniqueDepths = new Set(depths);
      
      // Use dark mode state if provided, otherwise detect from document
      const darkMode = isDarkMode !== undefined 
        ? isDarkMode 
        : document.documentElement.getAttribute('data-theme') === 'dark';
      
      // If all depths are the same or very few unique depths, show a simpler legend
      if (uniqueDepths.size <= 2 && maxDepth === 0) {
        // All models are root - show library-based legend instead
        const categories = Array.from(new Set(data.map((d: any) => d.library_name || 'unknown')));
        const colorMap = getCategoricalColorMap(categories, 'library');
        const topCategories = categories.slice(0, 10);
        items = topCategories.map(cat => ({
          label: cat.length > 20 ? cat.substring(0, 17) + '...' : cat,
          color: colorMap.get(cat) || '#4a90e2'
        }));
        isCont = false;
        isCat = true;
      } else {
        const scale = getDepthColorScale(maxDepth, darkMode);
        
        // Create gradient legend showing depth progression
        const steps = 8;
        for (let i = 0; i <= steps; i++) {
          const depth = (i / steps) * maxDepth;
          let label = '';
          if (i === 0) {
            label = `${minDepth} (Root)`;
          } else if (i === steps) {
            label = `${Math.round(maxDepth)} (Deep)`;
          } else if (i === Math.floor(steps / 2)) {
            label = `${Math.round(depth)}`;
          }
          items.push({
            label,
            color: scale(depth)
          });
        }
      }
    } else if (colorBy === 'downloads' || colorBy === 'likes') {
      isCont = true;
      const values = data.map((d: any) => {
        if (colorBy === 'downloads') return d.downloads;
        return d.likes;
      });
      const min = Math.min(...values);
      const max = Math.max(...values);
      const scale = getContinuousColorScale(min, max, 'viridis', true); // Use log scale
      
      // Create gradient legend
      const steps = 8;
      for (let i = 0; i <= steps; i++) {
        const logMin = Math.log10(min + 1);
        const logMax = Math.log10(max + 1);
        const logValue = logMin + (i / steps) * (logMax - logMin);
        const value = Math.pow(10, logValue) - 1;
        
        let label = '';
        if (i === 0) {
          label = min >= 1000 ? `${(min / 1000).toFixed(1)}K` : Math.round(min).toString();
        } else if (i === steps) {
          label = max >= 1000000 ? `${(max / 1000000).toFixed(1)}M` : max >= 1000 ? `${(max / 1000).toFixed(1)}K` : Math.round(max).toString();
        }
        
        items.push({
          label,
          color: scale(value)
        });
      }
    } else if (colorBy === 'trending_score') {
      isCont = true;
      const scores = data.map((d: any) => d.trending_score ?? 0);
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      const scale = getContinuousColorScale(min, max, 'plasma', false);
      
      const steps = 8;
      for (let i = 0; i <= steps; i++) {
        const value = min + (i / steps) * (max - min);
        items.push({
          label: i === 0 ? min.toFixed(1) : i === steps ? max.toFixed(1) : '',
          color: scale(value)
        });
      }
    }
    
    return { 
      legendItems: items, 
      isContinuous: isCont, 
      isCategorical: isCat,
      positionClass: `legend-${position}`
    };
  }, [data, colorBy, position, isDarkMode]); // Only recalculate when data, colorBy, position, or isDarkMode changes

  const getTitle = useMemo(() => {
    const titles: Record<string, string> = {
      'library_name': 'Library',
      'pipeline_tag': 'Pipeline / Task',
      'cluster_id': 'Cluster',
      'family_depth': 'Family Depth',
      'downloads': 'Downloads',
      'likes': 'Likes',
      'trending_score': 'Trending Score',
      'licenses': 'License'
    };
    return titles[colorBy] || colorBy.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }, [colorBy]);

  if (!data || data.length === 0 || legendItems.length === 0) return null;

  return (
    <div className={`color-legend ${positionClass}`} style={{ width }}>
      <div className="legend-header">
        <strong>{getTitle}</strong>
      </div>
      <div className="legend-content">
        {isCategorical ? (
          <div className="legend-categorical">
            {legendItems.map((item, idx) => (
              <div key={idx} className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="legend-label" title={item.label}>{item.label}</span>
              </div>
            ))}
            {legendItems.length >= 20 && (
              <div className="legend-item legend-item-more">
                <span className="legend-label">... and more</span>
              </div>
            )}
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
              <span>{legendItems[0].label || 'Min'}</span>
              <span>{legendItems[legendItems.length - 1].label || 'Max'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Memoize the component to prevent unnecessary re-renders
export default React.memo(ColorLegend, (prevProps, nextProps) => {
  // Only re-render if colorBy changes or data length changes significantly
  return prevProps.colorBy === nextProps.colorBy && 
         prevProps.position === nextProps.position &&
         prevProps.width === nextProps.width &&
         Math.abs(prevProps.data.length - nextProps.data.length) < 100; // Allow small fluctuations
});
