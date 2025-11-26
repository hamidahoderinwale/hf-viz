/**
 * Button to select a random model from the dataset for discovery.
 */
import React from 'react';
import { ModelPoint } from '../../types';

interface RandomModelButtonProps {
  data: ModelPoint[];
  onSelect: (model: ModelPoint) => void;
  disabled?: boolean;
}

export default function RandomModelButton({ data, onSelect, disabled }: RandomModelButtonProps) {
  const handleRandomSelect = () => {
    if (data.length === 0) return;
    const randomIndex = Math.floor(Math.random() * data.length);
    onSelect(data[randomIndex]);
  };

  return (
    <button
      onClick={handleRandomSelect}
      disabled={disabled || data.length === 0}
      className="random-model-btn"
      title="Select a random model"
      aria-label="Select random model"
    >
      <span>Select Random Model</span>
    </button>
  );
}

