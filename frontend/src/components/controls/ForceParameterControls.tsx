import React, { useState } from 'react';
import { Settings } from 'lucide-react';
import './ForceParameterControls.css';

interface ForceParameterControlsProps {
  linkDistance: number;
  chargeStrength: number;
  collisionRadius: number;
  nodeSizeMultiplier: number;
  edgeOpacity: number;
  onLinkDistanceChange: (value: number) => void;
  onChargeStrengthChange: (value: number) => void;
  onCollisionRadiusChange: (value: number) => void;
  onNodeSizeMultiplierChange: (value: number) => void;
  onEdgeOpacityChange: (value: number) => void;
}

export default function ForceParameterControls({
  linkDistance,
  chargeStrength,
  collisionRadius,
  nodeSizeMultiplier,
  edgeOpacity,
  onLinkDistanceChange,
  onChargeStrengthChange,
  onCollisionRadiusChange,
  onNodeSizeMultiplierChange,
  onEdgeOpacityChange,
}: ForceParameterControlsProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="force-parameter-controls">
      <button
        className="force-parameter-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        title="Force simulation parameters"
      >
        <Settings size={14} />
        <span>Parameters</span>
      </button>
      
      {isExpanded && (
        <div className="force-parameter-panel">
          <div className="force-parameter-group">
            <label>
              Link Distance: {linkDistance}
              <input
                type="range"
                min="50"
                max="200"
                step="10"
                value={linkDistance}
                onChange={(e) => onLinkDistanceChange(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="force-parameter-group">
            <label>
              Charge Strength: {chargeStrength}
              <input
                type="range"
                min="-500"
                max="-100"
                step="50"
                value={chargeStrength}
                onChange={(e) => onChargeStrengthChange(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="force-parameter-group">
            <label>
              Collision Radius: {collisionRadius.toFixed(1)}x
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={collisionRadius}
                onChange={(e) => onCollisionRadiusChange(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="force-parameter-group">
            <label>
              Node Size: {nodeSizeMultiplier.toFixed(1)}x
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={nodeSizeMultiplier}
                onChange={(e) => onNodeSizeMultiplierChange(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="force-parameter-group">
            <label>
              Edge Opacity: {edgeOpacity.toFixed(1)}
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={edgeOpacity}
                onChange={(e) => onEdgeOpacityChange(Number(e.target.value))}
              />
            </label>
          </div>
        </div>
      )}
    </div>
  );
}


