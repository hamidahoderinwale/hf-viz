import React from 'react';
import { EdgeType } from '../visualizations/ForceDirectedGraph';
import './EdgeTypeFilter.css';

interface EdgeTypeFilterProps {
  edgeTypes: EdgeType[];
  enabledTypes: Set<EdgeType>;
  onToggle: (type: EdgeType) => void;
  compact?: boolean;
}

const EDGE_COLORS: Record<EdgeType, string> = {
  finetune: '#3b82f6',
  quantized: '#10b981',
  adapter: '#f59e0b',
  merge: '#8b5cf6',
  parent: '#6b7280',
};

const EDGE_LABELS: Record<EdgeType, string> = {
  finetune: 'Fine-tuned',
  quantized: 'Quantized',
  adapter: 'Adapter',
  merge: 'Merged',
  parent: 'Parent',
};

export default function EdgeTypeFilter({ 
  edgeTypes, 
  enabledTypes, 
  onToggle,
  compact = false 
}: EdgeTypeFilterProps) {
  if (compact) {
    return (
      <div className="edge-type-filter-compact">
        {edgeTypes.map((type) => (
          <button
            key={type}
            className={`edge-type-toggle ${enabledTypes.has(type) ? 'active' : ''}`}
            onClick={() => onToggle(type)}
            title={EDGE_LABELS[type]}
            style={{
              backgroundColor: enabledTypes.has(type) ? EDGE_COLORS[type] : 'transparent',
              borderColor: EDGE_COLORS[type],
            }}
          >
            <span className="edge-type-toggle-label">{EDGE_LABELS[type]}</span>
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="edge-type-filter">
      <h4>Relationship Types</h4>
      {edgeTypes.map((type) => (
        <div
          key={type}
          className={`edge-type-item ${!enabledTypes.has(type) ? 'disabled' : ''}`}
          onClick={() => onToggle(type)}
        >
          <div
            className="edge-type-color"
            style={{ backgroundColor: EDGE_COLORS[type] }}
          />
          <span className="edge-type-label">{EDGE_LABELS[type]}</span>
        </div>
      ))}
    </div>
  );
}


