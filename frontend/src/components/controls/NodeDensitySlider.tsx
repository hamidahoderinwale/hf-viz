/**
 * Node density slider for controlling rendering performance.
 * Lower density improves performance for large datasets.
 */
import React from 'react';
import { useFilterStore } from '../../stores/filterStore';
import './NodeDensitySlider.css';

interface NodeDensitySliderProps {
  disabled?: boolean;
}

export default function NodeDensitySlider({ disabled = false }: NodeDensitySliderProps) {
  const { nodeDensity, setNodeDensity } = useFilterStore();

  return (
    <div className="node-density-slider">
      <label className="node-density-label">
        <span className="node-density-title">
          Node Density ({nodeDensity}%)
        </span>
        <input
          type="range"
          min="10"
          max="100"
          step="10"
          value={nodeDensity}
          onChange={(e) => setNodeDensity(parseInt(e.target.value))}
          disabled={disabled}
          className="node-density-input"
        />
        <div className="node-density-hint">
          Lower density improves performance for large datasets
        </div>
      </label>
    </div>
  );
}

