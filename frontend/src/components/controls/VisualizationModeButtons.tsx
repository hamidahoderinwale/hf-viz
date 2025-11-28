/**
 * Visualization mode buttons with sticky header.
 * Inspired by LAION's mode selection UI.
 */
import React from 'react';
import { useFilterStore, ViewMode } from '../../stores/filterStore';
import './VisualizationModeButtons.css';

interface ModeOption {
  value: ViewMode;
  label: string;
  icon: string;
  description: string;
}

const MODES: ModeOption[] = [
  { value: 'scatter', label: '2D Scatter', icon: '', description: '2D projection view' },
  { value: 'network', label: 'Network', icon: '', description: 'Network graph view' },
  { value: 'distribution', label: 'Distribution', icon: '', description: 'Statistical distributions' },
];

export default function VisualizationModeButtons() {
  const { viewMode, setViewMode } = useFilterStore();

  return (
    <div className="visualization-mode-buttons">
      <div className="mode-buttons-container">
        {MODES.map(mode => (
          <button
            key={mode.value}
            className={`mode-button ${viewMode === mode.value ? 'active' : ''}`}
            onClick={() => setViewMode(mode.value)}
            title={mode.description}
          >
            <span className="mode-icon">{mode.icon}</span>
            <span className="mode-label">{mode.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

