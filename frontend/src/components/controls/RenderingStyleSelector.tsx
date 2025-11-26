/**
 * Rendering style selector dropdown.
 * Allows users to choose different 3D layout/geometry styles.
 */
import React from 'react';
import { useFilterStore, RenderingStyle } from '../../stores/filterStore';
import './RenderingStyleSelector.css';

const STYLES: { value: RenderingStyle; label: string; description: string }[] = [
  { value: 'embeddings', label: 'Embeddings', description: 'Standard embedding-based layout' },
  { value: 'sphere', label: 'Sphere', description: 'Spherical arrangement of points' },
  { value: 'galaxy', label: 'Galaxy', description: 'Spiral galaxy-like layout' },
  { value: 'wave', label: 'Wave', description: 'Wave pattern arrangement' },
  { value: 'helix', label: 'Helix', description: 'Helical/spiral arrangement' },
  { value: 'torus', label: 'Torus', description: 'Torus/donut-shaped layout' },
];

export default function RenderingStyleSelector() {
  const { renderingStyle, setRenderingStyle } = useFilterStore();

  return (
    <div className="rendering-style-selector">
      <label className="rendering-style-label">
        <span className="rendering-style-title">Rendering Style</span>
        <select
          value={renderingStyle}
          onChange={(e) => setRenderingStyle(e.target.value as RenderingStyle)}
          className="rendering-style-select"
        >
          {STYLES.map(style => (
            <option key={style.value} value={style.value}>
              {style.label}
            </option>
          ))}
        </select>
        <div className="rendering-style-hint">
          {STYLES.find(s => s.value === renderingStyle)?.description}
        </div>
      </label>
    </div>
  );
}

