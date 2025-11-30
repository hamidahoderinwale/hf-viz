import React from 'react';
import './LoadingProgress.css';

interface LoadingProgressProps {
  message?: string;
  progress?: number; // 0-100
  subMessage?: string;
}

export default function LoadingProgress({ 
  message = 'Loading models...', 
  progress,
  subMessage 
}: LoadingProgressProps) {
  return (
    <div className="loading-progress">
      <div className="loading-progress-content">
        <div className="loading-spinner" />
        <div className="loading-message">{message}</div>
        {subMessage && (
          <div className="loading-submessage">{subMessage}</div>
        )}
        {progress !== undefined && (
          <div className="loading-bar-container">
            <div 
              className="loading-bar" 
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

