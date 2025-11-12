/**
 * Modal component for displaying detailed model information.
 */
import React from 'react';
import { ModelPoint } from '../types';
import './ModelModal.css';

interface ModelModalProps {
  model: ModelPoint | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function ModelModal({ model, isOpen, onClose }: ModelModalProps) {
  if (!isOpen || !model) return null;

  const hfUrl = `https://huggingface.co/${model.model_id}`;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <h2>{model.model_id}</h2>
        
        <div className="modal-section">
          <h3>Model Information</h3>
          <div className="modal-info-grid">
            <div className="modal-info-item">
              <strong>Library:</strong>
              <span>{model.library_name || 'N/A'}</span>
            </div>
            <div className="modal-info-item">
              <strong>Pipeline Tag:</strong>
              <span>{model.pipeline_tag || 'N/A'}</span>
            </div>
            <div className="modal-info-item">
              <strong>Downloads:</strong>
              <span>{model.downloads.toLocaleString()}</span>
            </div>
            <div className="modal-info-item">
              <strong>Likes:</strong>
              <span>{model.likes.toLocaleString()}</span>
            </div>
            {model.trending_score !== null && (
              <div className="modal-info-item">
                <strong>Trending Score:</strong>
                <span>{model.trending_score.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>

        {model.tags && (
          <div className="modal-section">
            <h3>Tags</h3>
            <p className="modal-tags">{model.tags}</p>
          </div>
        )}

        <div className="modal-section">
          <h3>Links</h3>
          <a 
            href={hfUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="modal-link"
          >
            View on Hugging Face →
          </a>
        </div>

        <div className="modal-section">
          <h3>Position in Latent Space</h3>
          <div className="modal-info-grid">
            <div className="modal-info-item">
              <strong>Dimension 1:</strong>
              <span>{model.x.toFixed(4)}</span>
            </div>
            <div className="modal-info-item">
              <strong>Dimension 2:</strong>
              <span>{model.y.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

