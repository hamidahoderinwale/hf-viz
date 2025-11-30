/**
 * Tooltip component for displaying model information when hovering over points in 3D plot
 */
import React, { useEffect, useState } from 'react';
import { ModelPoint } from '../../types';
import { getHuggingFaceApiUrl } from '../../utils/api/hfUrl';
import './ModelTooltip.css';

interface ModelTooltipProps {
  model: ModelPoint | null;
  position: { x: number; y: number } | null;
  visible: boolean;
}

interface ModelDetails {
  description?: string;
  loading?: boolean;
  error?: string;
}

function formatDate(dateString: string | null): string {
  if (!dateString) return '';
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  } catch {
    return dateString;
  }
}

export default function ModelTooltip({ model, position, visible }: ModelTooltipProps) {
  const [details, setDetails] = useState<ModelDetails>({});
  const [cache, setCache] = useState<Map<string, string>>(new Map());

  useEffect(() => {
    if (!model || !visible) {
      setDetails({});
      return;
    }

    if (cache.has(model.model_id)) {
      setDetails({ description: cache.get(model.model_id) });
      return;
    }

    setDetails({ loading: true });
    
    const fetchDescription = async () => {
      try {
        const hfToken = process.env.REACT_APP_HF_TOKEN || 
                       (typeof window !== 'undefined' ? localStorage.getItem('HF_TOKEN') : null);
        
        const headers: HeadersInit = {
          'Accept': 'application/json',
        };
        
        if (hfToken) {
          headers['Authorization'] = `Bearer ${hfToken}`;
        }
        
        const response = await fetch(getHuggingFaceApiUrl(model.model_id), {
          headers,
        });
        
        if (response.ok) {
          const data = await response.json();
          const description = data.cardData?.model_index?.general?.description || 
                            data.cardData?.model_index?.model?.description ||
                            data.siblings?.find((s: any) => s.rfilename === 'README.md')?.description ||
                            null;
          
          if (description) {
            const newCache = new Map(cache);
            newCache.set(model.model_id, description);
            setCache(newCache);
            setDetails({ description });
          } else {
            setDetails({});
          }
        } else {
          setDetails({});
        }
      } catch {
        setDetails({});
      }
    };

    fetchDescription();
  }, [model?.model_id, visible, cache]);

  if (!visible || !model || !position) {
    return null;
  }

  const description = details.description || '';
  const truncatedDescription = description.length > 200 
    ? description.substring(0, 200) + '...' 
    : description;

  return (
    <div
      className="model-tooltip"
      style={{
        left: `${position.x + 15}px`,
        top: `${position.y - 10}px`,
      }}
    >
      <div className="model-tooltip-title">
        {model.model_id}
      </div>
      
      <div className="model-tooltip-content">
        {model.library_name && (
          <div className="model-tooltip-row">
            <span className="model-tooltip-label">Library:</span> {model.library_name}
          </div>
        )}
        {model.pipeline_tag && (
          <div className="model-tooltip-row">
            <span className="model-tooltip-label">Task:</span> {model.pipeline_tag}
          </div>
        )}
        <div className="model-tooltip-row">
          <span className="model-tooltip-label">Downloads:</span> {model.downloads.toLocaleString()} | 
          <span className="model-tooltip-label-spaced">Likes:</span> {model.likes.toLocaleString()}
        </div>
        {model.created_at && (
          <div className="model-tooltip-row-small">
            <span className="model-tooltip-label">Created:</span> {formatDate(model.created_at)}
          </div>
        )}
        {model.parent_model && (
          <div className="model-tooltip-row-small">
            <span className="model-tooltip-label">Parent:</span> {model.parent_model}
          </div>
        )}
      </div>

      {details.loading && (
        <div className="model-tooltip-loading">
          Loading description...
        </div>
      )}

      {truncatedDescription && (
        <div className="model-tooltip-description">
          {truncatedDescription}
        </div>
      )}

      <div className="model-tooltip-hint">
        Click for details
      </div>
    </div>
  );
}
