/**
 * Modal component for displaying detailed model information.
 * Enhanced with bookmark, comparison, similar models, and file tree features.
 */
import React, { useState, useEffect } from 'react';
import { ModelPoint } from '../../types';
import FileTree from './FileTree';
import { getHuggingFaceUrl } from '../../utils/api/hfUrl';
import { API_BASE } from '../../config/api';
import './ModelModal.css';

interface ArxivPaper {
  arxiv_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  categories: string[];
  url: string;
}

interface ModelModalProps {
  model: ModelPoint | null;
  isOpen: boolean;
  onClose: () => void;
  onBookmark?: (modelId: string) => void;
  onAddToComparison?: (model: ModelPoint) => void;
  onLoadSimilar?: (modelId: string) => void;
  isBookmarked?: boolean;
}

export default function ModelModal({
  model,
  isOpen,
  onClose,
  onBookmark,
  onAddToComparison,
  onLoadSimilar,
  isBookmarked = false,
}: ModelModalProps) {
  const [activeTab, setActiveTab] = useState<'details' | 'files' | 'papers'>('details');
  const [papers, setPapers] = useState<ArxivPaper[]>([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [papersError, setPapersError] = useState<string | null>(null);

  // Fetch arXiv papers when model changes
  useEffect(() => {
    if (!isOpen || !model) {
      setPapers([]);
      return;
    }

    const fetchPapers = async () => {
      setPapersLoading(true);
      setPapersError(null);
      try {
        const response = await fetch(`${API_BASE}/api/model/${encodeURIComponent(model.model_id)}/papers`);
        if (!response.ok) throw new Error('Failed to fetch papers');
        const data = await response.json();
        setPapers(data.papers || []);
      } catch (err) {
        setPapersError(err instanceof Error ? err.message : 'Failed to load papers');
        setPapers([]);
      } finally {
        setPapersLoading(false);
      }
    };

    fetchPapers();
  }, [model?.model_id, isOpen]);

  if (!isOpen || !model) return null;

  const hfUrl = getHuggingFaceUrl(model.model_id);

  // Parse tags if it's a string representation of an array
  const parseTags = (tags: string | null | undefined): string[] => {
    if (!tags) return [];
    try {
      // Try to parse as JSON array
      if (tags.startsWith('[') && tags.endsWith(']')) {
        // Replace single quotes with double quotes for valid JSON
        const jsonString = tags.replace(/'/g, '"');
        return JSON.parse(jsonString);
      }
      // Otherwise split by comma
      return tags.split(',').map(t => t.trim().replace(/['"]/g, ''));
    } catch {
      // If parsing fails, try to extract values from string representation
      try {
        // Handle cases like "['tag1', 'tag2']"
        const cleaned = tags.replace(/^\[|\]$/g, '').replace(/'/g, '"');
        const parsed = JSON.parse(`[${cleaned}]`);
        return Array.isArray(parsed) ? parsed : [tags];
      } catch {
        return [tags];
      }
    }
  };

  const tags = parseTags(model.tags);
  
  // Parse licenses - handle both JSON arrays and string representations with single quotes
  const parseLicenses = (licenses: string | null | undefined): string[] => {
    if (!licenses) return [];
    try {
      // If it's already a valid JSON array string, parse it
      if (licenses.startsWith('[') && licenses.endsWith(']')) {
        // Replace single quotes with double quotes for valid JSON
        const jsonString = licenses.replace(/'/g, '"');
        return JSON.parse(jsonString);
      }
      // Otherwise, treat as a single license string
      return [licenses];
    } catch {
      // If parsing fails, try to extract values from string representation
      try {
        // Handle cases like "['apache-2.0']" or "['license1', 'license2']"
        const cleaned = licenses.replace(/^\[|\]$/g, '').replace(/'/g, '"');
        const parsed = JSON.parse(`[${cleaned}]`);
        return Array.isArray(parsed) ? parsed : [licenses];
      } catch {
        // Last resort: return as single-item array
        return [licenses];
      }
    }
  };
  
  const licenses = parseLicenses(model.licenses);

  // Color coding functions
  const getLibraryColor = (library: string | null | undefined): string => {
    if (!library) return '#cccccc';
    const colors: Record<string, string> = {
      'transformers': '#1f77b4',
      'diffusers': '#ff7f0e',
      'sentence-transformers': '#2ca02c',
      'timm': '#d62728',
      'speechbrain': '#9467bd',
    };
    return colors[library.toLowerCase()] || '#6a6a6a';
  };

  const getPipelineColor = (pipeline: string | null | undefined): string => {
    if (!pipeline || pipeline === 'Unknown') return '#cccccc';
    const colors: Record<string, string> = {
      'text-classification': '#1f77b4',
      'token-classification': '#ff7f0e',
      'question-answering': '#2ca02c',
      'summarization': '#d62728',
      'translation': '#9467bd',
      'text-generation': '#8c564b',
    };
    return colors[pipeline.toLowerCase()] || '#6a6a6a';
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div 
        className="modal-content" 
        onClick={(e) => e.stopPropagation()}
        data-tab={activeTab}
      >
        <div className="modal-header">
          <h2>{model.model_id}</h2>
          <button className="modal-close" onClick={onClose}>Close</button>
        </div>

        <div className="modal-body">
          {/* Action Buttons */}
          <div className="modal-actions">
            {onBookmark && (
              <button
                onClick={() => onBookmark(model.model_id)}
                className={`action-btn ${isBookmarked ? 'active' : ''}`}
              >
                {isBookmarked ? '✓ Bookmarked' : 'Bookmark'}
              </button>
            )}
            {onAddToComparison && (
              <button
                onClick={() => onAddToComparison(model)}
                className="action-btn"
              >
                Add to Comparison
              </button>
            )}
            {onLoadSimilar && (
              <button
                onClick={() => onLoadSimilar(model.model_id)}
                className="action-btn"
              >
                Find Similar Models
              </button>
            )}
          </div>

          {/* Tabs */}
          <div className="modal-tabs">
            <button
              className={`modal-tab ${activeTab === 'details' ? 'active' : ''}`}
              onClick={() => setActiveTab('details')}
            >
              <span className="tab-icon"></span>
              <span>Details</span>
            </button>
            <button
              className={`modal-tab ${activeTab === 'files' ? 'active' : ''}`}
              onClick={() => setActiveTab('files')}
            >
              <span className="tab-icon"></span>
              <span>Files</span>
            </button>
            {(papers.length > 0 || papersLoading) && (
              <button
                className={`modal-tab ${activeTab === 'papers' ? 'active' : ''}`}
                onClick={() => setActiveTab('papers')}
              >
                <span className="tab-icon"></span>
                <span>Papers</span>
                {papers.length > 0 && <span className="tab-badge">{papers.length}</span>}
              </button>
            )}
          </div>

          {/* Tab Content */}
          {activeTab === 'details' && (
            <div className="modal-info-section">
              <div className="info-grid">
                <div className="info-item">
                  <div className="info-label">Library</div>
                  <div 
                    className="info-value colored"
                    style={{ 
                      color: getLibraryColor(model.library_name),
                      fontWeight: 600 
                    }}
                  >
                    {model.library_name || 'Unknown'}
                  </div>
                </div>

                <div className="info-item">
                  <div className="info-label">Pipeline / Task</div>
                  <div 
                    className="info-value colored"
                    style={{ 
                      color: getPipelineColor(model.pipeline_tag),
                      fontWeight: 600 
                    }}
                  >
                    {model.pipeline_tag || 'Unknown'}
                  </div>
                </div>

                <div className="info-item">
                  <div className="info-label">Downloads</div>
                  <div className="info-value highlight">
                    {model.downloads.toLocaleString()}
                  </div>
                </div>

                <div className="info-item">
                  <div className="info-label">Likes</div>
                  <div className="info-value highlight">
                    {model.likes.toLocaleString()}
                  </div>
                </div>

                {model.trending_score !== null && (
                  <div className="info-item">
                    <div className="info-label">Trending Score</div>
                    <div className="info-value">
                      {model.trending_score.toFixed(2)}
                    </div>
                  </div>
                )}

                <div className="info-item">
                  <div className="info-label">Coordinates</div>
                  <div className="info-value coordinates">
                    <span>X: {model.x.toFixed(3)}</span>
                    <span>Y: {model.y.toFixed(3)}</span>
                    <span>Z: {model.z.toFixed(3)}</span>
                  </div>
                </div>
              </div>

              {model.parent_model && (
                <div className="info-section">
                  <div className="section-title">Parent Model</div>
                  <div className="section-content">
                    <a
                      href={getHuggingFaceUrl(model.parent_model)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="model-link"
                    >
                      {model.parent_model}
                    </a>
                  </div>
                </div>
              )}

              {licenses.length > 0 && (
                <div className="info-section">
                  <div className="section-title">License{licenses.length > 1 ? 's' : ''}</div>
                  <div className="section-content">
                    <div className="tag-list">
                      {licenses.map((license: string, idx: number) => (
                        <span key={idx} className="tag license-tag">
                          {license}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {tags.length > 0 && (
                <div className="info-section">
                  <div className="section-title">Tags</div>
                  <div className="section-content">
                    <div className="tag-list">
                      {tags.map((tag: string, idx: number) => (
                        <span key={idx} className="tag">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'files' && (
            <div className="modal-info-section">
              <FileTree modelId={model.model_id} />
            </div>
          )}

          {activeTab === 'papers' && (
            <div className="modal-info-section">
              {papersLoading ? (
                <div className="papers-loading">Loading papers...</div>
              ) : papersError ? (
                <div className="papers-error">Error loading papers: {papersError}</div>
              ) : papers.length === 0 ? (
                <div className="papers-empty">No arXiv papers found for this model.</div>
              ) : (
                <div className="papers-list">
                  {papers.map((paper, idx) => (
                    <div key={paper.arxiv_id || idx} className="paper-card">
                      <div className="paper-header">
                        <h3 className="paper-title">
                          <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="paper-link"
                          >
                            {paper.title}
                          </a>
                        </h3>
                        <div className="paper-id">
                          <a
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="arxiv-link"
                          >
                            arXiv:{paper.arxiv_id}
                          </a>
                        </div>
                      </div>
                      
                      {paper.authors && paper.authors.length > 0 && (
                        <div className="paper-authors">
                          <strong>Authors:</strong> {paper.authors.join(', ')}
                        </div>
                      )}
                      
                      {paper.published && (
                        <div className="paper-date">
                          <strong>Published:</strong> {new Date(paper.published).toLocaleDateString()}
                        </div>
                      )}
                      
                      {paper.categories && paper.categories.length > 0 && (
                        <div className="paper-categories">
                          <strong>Categories:</strong>{' '}
                          {paper.categories.map((cat, i) => (
                            <span key={i} className="category-tag">
                              {cat}
                            </span>
                          ))}
                        </div>
                      )}
                      
                      {paper.abstract && (
                        <div className="paper-abstract">
                          <strong>Abstract:</strong>
                          <p>{paper.abstract}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="modal-footer">
            <a
              href={hfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="modal-link"
            >
              View on Hugging Face →
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
