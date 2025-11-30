/**
 * Popup component for displaying model information in the bottom left.
 * Replaces the full-screen modal with a compact, persistent popup.
 */
import React, { useState, useEffect } from 'react';
import { X, ArrowUpRight, Bookmark, Download, Heart, TrendingUp, GitBranch, Tag, Layers, Box } from 'lucide-react';
import { ModelPoint } from '../../types';
import { getHuggingFaceUrl } from '../../utils/api/hfUrl';
import { API_BASE } from '../../config/api';
import './ModelPopup.css';

interface ModelPopupProps {
  model: ModelPoint | null;
  isOpen: boolean;
  onClose: () => void;
  onBookmark?: (modelId: string) => void;
  isBookmarked?: boolean;
}

export default function ModelPopup({
  model,
  isOpen,
  onClose,
  onBookmark,
  isBookmarked = false,
}: ModelPopupProps) {
  const [lineagePath, setLineagePath] = useState<string[]>([]);
  const [lineageLoading, setLineageLoading] = useState(false);

  // Fetch lineage path when model changes
  useEffect(() => {
    if (!isOpen || !model) {
      setLineagePath([]);
      return;
    }

    const fetchLineage = async () => {
      setLineageLoading(true);
      try {
        const response = await fetch(`${API_BASE}/api/family/path/${encodeURIComponent(model.model_id)}`);
        if (response.ok) {
          const data = await response.json();
          setLineagePath(data.path || []);
        } else {
          setLineagePath([]);
        }
      } catch {
        setLineagePath([]);
      } finally {
        setLineageLoading(false);
      }
    };

    fetchLineage();
  }, [model?.model_id, isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !model) return null;

  const formatNumber = (num: number | null): string => {
    if (num === null || num === undefined) return '—';
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toLocaleString();
  };

  const formatDate = (dateString: string | null): string => {
    if (!dateString) return '—';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="model-popup">
      {/* Header */}
      <div className="popup-header">
        <div className="popup-title-row">
          <div className="popup-title-wrapper">
            <h3 className="popup-title" title={model.model_id}>
              {model.model_id}
            </h3>
            {!model.parent_model && (model.family_depth === 0 || model.family_depth === null) && (
              <span className="popup-base-badge" title="This is a base model with no parent">BASE</span>
            )}
          </div>
          <div className="popup-actions">
            {onBookmark && (
              <button
                className={`popup-bookmark-btn ${isBookmarked ? 'active' : ''}`}
                onClick={() => onBookmark(model.model_id)}
                title={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
              >
                <Bookmark size={16} fill={isBookmarked ? 'currentColor' : 'none'} />
              </button>
            )}
            <button className="popup-close-btn" onClick={onClose} title="Close">
              <X size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="popup-content">
        {/* Stats Row */}
        <div className="popup-stats">
          <div className="popup-stat" title={`Downloads: ${model.downloads?.toLocaleString() || 0} total downloads from Hugging Face`}>
            <Download size={14} />
            <span>{formatNumber(model.downloads)}</span>
          </div>
          <div className="popup-stat" title={`Likes: ${model.likes?.toLocaleString() || 0} community likes on Hugging Face`}>
            <Heart size={14} />
            <span>{formatNumber(model.likes)}</span>
          </div>
          {model.trending_score !== null && model.trending_score > 0 && (
            <div className="popup-stat trending" title={`Trending Score: ${model.trending_score.toFixed(2)} — measures recent popularity growth`}>
              <TrendingUp size={14} />
              <span>{model.trending_score.toFixed(1)}</span>
            </div>
          )}
        </div>

        {/* Info Grid */}
        <div className="popup-info-grid">
          {model.library_name && (
            <div className="popup-info-item" title={`ML Library: The framework used to build this model (e.g., transformers, diffusers, timm)`}>
              <Layers size={14} />
              <span className="popup-info-label">Library</span>
              <span className="popup-info-value">{model.library_name}</span>
            </div>
          )}
          {model.pipeline_tag && (
            <div className="popup-info-item" title={`Task Type: What this model does (e.g., text-generation, image-classification)`}>
              <Tag size={14} />
              <span className="popup-info-label">Task</span>
              <span className="popup-info-value">{model.pipeline_tag}</span>
            </div>
          )}
          {model.created_at && (
            <div className="popup-info-item" title="When this model was first published on Hugging Face">
              <span className="popup-info-label">Created</span>
              <span className="popup-info-value">{formatDate(model.created_at)}</span>
            </div>
          )}
          {model.family_depth !== null && model.family_depth !== undefined && (
            <div className="popup-info-item" title={`Family Depth: ${model.family_depth === 0 ? 'Base model (root of lineage tree)' : `${model.family_depth} generation${model.family_depth > 1 ? 's' : ''} from the root model`}`}>
              <GitBranch size={14} />
              <span className="popup-info-label">Depth</span>
              <span className="popup-info-value">{model.family_depth}</span>
            </div>
          )}
        </div>

        {/* Lineage */}
        <div className="popup-lineage" title="Model lineage shows the parent-child relationship from the original base model to this model">
          <div className="popup-lineage-label">Lineage</div>
          <div className="popup-lineage-path">
            {lineageLoading ? (
              <span className="popup-lineage-loading">Loading...</span>
            ) : lineagePath.length > 1 ? (
              lineagePath.map((pathModel, idx) => (
                <React.Fragment key={pathModel}>
                  <span 
                    className={`popup-lineage-item ${pathModel === model.model_id ? 'current' : ''}`}
                    title={pathModel}
                  >
                    {pathModel.split('/').pop()}
                  </span>
                  {idx < lineagePath.length - 1 && (
                    <span className="popup-lineage-separator">&gt;</span>
                  )}
                </React.Fragment>
              ))
            ) : model.parent_model ? (
              <>
                <span className="popup-lineage-item" title={model.parent_model}>
                  {model.parent_model.split('/').pop()}
                </span>
                <span className="popup-lineage-separator">&gt;</span>
                <span className="popup-lineage-item current" title={model.model_id}>
                  {model.model_id.split('/').pop()}
                </span>
              </>
            ) : (
              <span className="popup-lineage-base">
                <Box size={14} />
                Root model
              </span>
            )}
          </div>
        </div>

        {/* Tags */}
        {model.tags && (
          <div className="popup-tags">
            {(typeof model.tags === 'string' ? model.tags.split(',') : [])
              .slice(0, 6)
              .map((tag, idx) => (
                <span key={idx} className="popup-tag">{tag.trim()}</span>
              ))}
            {(typeof model.tags === 'string' ? model.tags.split(',') : []).length > 6 && (
              <span className="popup-tag-more">
                +{(typeof model.tags === 'string' ? model.tags.split(',') : []).length - 6}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="popup-footer">
        <a
          href={getHuggingFaceUrl(model.model_id)}
          target="_blank"
          rel="noopener noreferrer"
          className="popup-hf-link"
        >
          View on Hugging Face
          <ArrowUpRight size={14} />
        </a>
      </div>
    </div>
  );
}

