/**
 * Tooltip component for displaying model information when hovering over points in 3D plot
 */
import React, { useEffect, useState } from 'react';
import { ModelPoint } from '../../types';
import { getHuggingFaceApiUrl } from '../../utils/api/hfUrl';

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
    if (isNaN(date.getTime())) return dateString; // Return original if invalid
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

    // Check cache first
    if (cache.has(model.model_id)) {
      setDetails({ description: cache.get(model.model_id) });
      return;
    }

    // Fetch model description from Hugging Face API
    setDetails({ loading: true });
    
    const fetchDescription = async () => {
      try {
        // Try to get description from Hugging Face API
        // Use HF token if available (from env or localStorage)
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
            // Cache the description
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
      } catch (error) {
        console.error('Error fetching model description:', error);
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
      style={{
        position: 'fixed',
        left: `${position.x + 15}px`,
        top: `${position.y - 10}px`,
        background: 'rgba(0, 0, 0, 0.9)',
        color: 'white',
        padding: '12px 16px',
        borderRadius: '0',
        fontSize: '13px',
        maxWidth: '350px',
        zIndex: 10000,
        pointerEvents: 'none',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <div style={{ fontWeight: '600', marginBottom: '8px', fontSize: '14px', color: '#fff' }}>
        {model.model_id}
      </div>
      
      <div style={{ marginBottom: '6px', fontSize: '12px', color: '#d0d0d0' }}>
        {model.library_name && (
          <div style={{ marginBottom: '4px' }}>
            <span style={{ color: '#888' }}>Library:</span> {model.library_name}
          </div>
        )}
        {model.pipeline_tag && (
          <div style={{ marginBottom: '4px' }}>
            <span style={{ color: '#888' }}>Task:</span> {model.pipeline_tag}
          </div>
        )}
        <div style={{ marginBottom: '4px' }}>
          <span style={{ color: '#888' }}>Downloads:</span> {model.downloads.toLocaleString()} | 
          <span style={{ color: '#888', marginLeft: '8px' }}>Likes:</span> {model.likes.toLocaleString()}
        </div>
        {model.created_at && (
          <div style={{ marginBottom: '4px', fontSize: '11px', color: '#aaa' }}>
            <span style={{ color: '#888' }}>Created:</span> {formatDate(model.created_at)}
          </div>
        )}
        {model.parent_model && (
          <div style={{ marginBottom: '4px', fontSize: '11px', color: '#aaa' }}>
            <span style={{ color: '#888' }}>Parent:</span> {model.parent_model}
          </div>
        )}
      </div>

      {details.loading && (
        <div style={{ fontSize: '11px', color: '#888', fontStyle: 'italic', marginTop: '8px' }}>
          Loading description...
        </div>
      )}

      {truncatedDescription && (
        <div style={{ 
          marginTop: '8px', 
          paddingTop: '8px', 
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '12px',
          color: '#e0e0e0',
          lineHeight: '1.4',
        }}>
          {truncatedDescription}
        </div>
      )}

      <div style={{ 
        marginTop: '8px', 
        fontSize: '11px', 
        color: '#888',
        fontStyle: 'italic' 
      }}>
        Click for details
      </div>
    </div>
  );
}

