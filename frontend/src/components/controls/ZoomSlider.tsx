/**
 * Slider control for zoom level in 3D visualization.
 */
import React from 'react';

interface ZoomSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

export default function ZoomSlider({
  value,
  onChange,
  min = 0.1,
  max = 5,
  step = 0.1,
  disabled = false,
}: ZoomSliderProps) {
  return (
    <div className="zoom-slider-container">
      <label className="zoom-slider-label">
        <span>Zoom Level</span>
        <span className="zoom-value">{value.toFixed(1)}x</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="zoom-slider"
        aria-label="Zoom level"
      />
    </div>
  );
}

