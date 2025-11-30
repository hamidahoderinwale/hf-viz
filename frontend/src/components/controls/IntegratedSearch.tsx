import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X, Search, ArrowRight, Download, Heart, Sparkles } from 'lucide-react';
import { API_BASE } from '../../config/api';
import './IntegratedSearch.css';

interface SearchResult {
  model_id: string;
  x?: number;
  y?: number;
  z?: number;
  similarity?: number;
  family_depth?: number | null;
  downloads?: number;
  likes?: number;
  library_name?: string | null;
  pipeline_tag?: string | null;
  // Fuzzy search additions
  score?: number;
}

interface IntegratedSearchProps {
  value: string;
  onChange: (value: string) => void;
  onSelect?: (result: SearchResult) => void;
  onZoomTo?: (x: number, y: number, z: number) => void;
}

export default function IntegratedSearch({
  value,
  onChange,
  onSelect,
  onZoomTo
}: IntegratedSearchProps) {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [searchType, setSearchType] = useState<'fuzzy' | 'semantic'>('fuzzy');
  const [localQuery, setLocalQuery] = useState('');
  
  const modalInputRef = useRef<HTMLInputElement>(null);
  const triggerInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Detect if query looks like a model ID (contains "/")
  const isModelId = useCallback((query: string): boolean => {
    return query.includes('/') && query.length >= 3;
  }, []);

  // Perform fuzzy search via API (searches all 2M+ models)
  const performFuzzySearch = useCallback(async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setResults([]);
      setTotalMatches(0);
      return;
    }

    setIsLoading(true);
    
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        limit: '100',
        threshold: '50', // 50% minimum match score
      });

      const response = await fetch(`${API_BASE}/api/search/fuzzy?${params}`);
      
      if (!response.ok) {
        throw new Error('Fuzzy search failed');
      }
      
      const data = await response.json();
      const models = (data.results || []).map((m: any) => ({
        model_id: m.model_id,
        x: m.x || 0,
        y: m.y || 0,
        z: m.z || 0,
        downloads: m.downloads || 0,
        likes: m.likes || 0,
        library_name: m.library || null,
        pipeline_tag: m.pipeline || null,
        family_depth: m.family_depth || null,
        score: m.score, // Server-side fuzzy score (0-100)
      }));

      setResults(models);
      setTotalMatches(data.total_matches || models.length);
      setSelectedIndex(-1);
    } catch (error) {
      console.error('Fuzzy search error:', error);
      setResults([]);
      setTotalMatches(0);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Perform semantic search for model IDs
  const performSemanticSearch = useCallback(async (queryModelId: string) => {
    if (!queryModelId || queryModelId.length < 3 || !queryModelId.includes('/')) {
      setResults([]);
      setTotalMatches(0);
      setIsLoading(false);
      return;
    }
    
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        query_model_id: queryModelId,
        k: '50',
        min_downloads: '0',
        min_likes: '0',
        projection_method: 'umap',
      });

      const response = await fetch(`${API_BASE}/api/models/semantic-similarity?${params}`);
      
      if (response.status === 404) {
        // Model not found, fall back to fuzzy search
        setSearchType('fuzzy');
        await performFuzzySearch(queryModelId);
        return;
      }
      
      if (!response.ok) {
        throw new Error('Semantic search failed');
      }
      
      const data = await response.json();
      const models = (data.models || []).map((m: any) => ({
        model_id: m.model_id,
        x: m.x,
        y: m.y,
        z: m.z,
        similarity: m.similarity,
        downloads: m.downloads,
        likes: m.likes,
        library_name: m.library_name,
        pipeline_tag: m.pipeline_tag,
        family_depth: m.family_depth,
      }));

      // Sort by similarity (highest first)
      models.sort((a: SearchResult, b: SearchResult) => 
        (b.similarity || 0) - (a.similarity || 0)
      );

      setResults(models.slice(0, 50));
      setTotalMatches(models.length);
      setSelectedIndex(-1);
    } catch {
      // Fall back to fuzzy search on error
      setSearchType('fuzzy');
      await performFuzzySearch(queryModelId);
    } finally {
      setIsLoading(false);
    }
  }, [performFuzzySearch]);

  // Handle search when local query changes
  useEffect(() => {
    if (localQuery.length < 2) {
      setResults([]);
      return;
    }

    const timer = setTimeout(() => {
      // Auto-detect search type: if it looks like a model ID, use semantic search
      if (isModelId(localQuery)) {
        setSearchType('semantic');
        performSemanticSearch(localQuery);
      } else {
        setSearchType('fuzzy');
        performFuzzySearch(localQuery);
      }
    }, 150); // Faster debounce for fuzzy search

    return () => clearTimeout(timer);
  }, [localQuery, isModelId, performSemanticSearch, performFuzzySearch]);

  const handleSelect = useCallback((result: SearchResult) => {
    if (onZoomTo && result.x !== undefined && result.y !== undefined) {
      onZoomTo(result.x, result.y, result.z || 0);
    }
    
    if (onSelect) {
      onSelect(result);
    }
    
    setIsModalOpen(false);
    setLocalQuery('');
    onChange('');
  }, [onSelect, onZoomTo, onChange]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (results.length === 0) return;
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < results.length - 1 ? prev + 1 : prev
      );
      // Scroll selected item into view
      if (resultsRef.current) {
        const items = resultsRef.current.querySelectorAll('.search-result-item');
        const nextIndex = selectedIndex < results.length - 1 ? selectedIndex + 1 : selectedIndex;
        items[nextIndex]?.scrollIntoView({ block: 'nearest' });
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
      if (resultsRef.current && selectedIndex > 0) {
        const items = resultsRef.current.querySelectorAll('.search-result-item');
        items[selectedIndex - 1]?.scrollIntoView({ block: 'nearest' });
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && results[selectedIndex]) {
        handleSelect(results[selectedIndex]);
      } else if (results.length > 0) {
        handleSelect(results[0]);
      }
    } else if (e.key === 'Escape') {
      setIsModalOpen(false);
    }
  };

  const openModal = () => {
    setIsModalOpen(true);
    setLocalQuery(value);
    setTimeout(() => modalInputRef.current?.focus(), 50);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setLocalQuery('');
    setResults([]);
  };

  // Handle Cmd/Ctrl+K shortcut
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        openModal();
      }
    };

    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => document.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  // Format relevance score (server returns 0-100)
  const formatScore = (score?: number) => {
    if (score === undefined) return null;
    return `${Math.round(score)}%`;
  };

  return (
    <>
      {/* Trigger input in control bar */}
      <div className="integrated-search-container">
        <div className="control-search" onClick={openModal}>
          <Search size={14} className="search-icon" />
          <input
            ref={triggerInputRef}
            type="text"
            value={value}
            readOnly
            onClick={openModal}
            onFocus={openModal}
            placeholder="Search models, tags, or model ID..."
            className="control-search-input"
            aria-label="Search models"
            title="Fuzzy search: finds models even with typos. Enter a model ID (e.g., meta-llama/Llama-2-7b) to find similar models."
          />
          <span className="search-shortcut">
            <kbd>Cmd</kbd>
            <kbd>K</kbd>
          </span>
        </div>
      </div>

      {/* Search Modal */}
      {isModalOpen && (
        <div className="search-modal-overlay" onClick={closeModal}>
          <div className="search-modal" onClick={(e) => e.stopPropagation()}>
            <div className="search-modal-header">
              <div className="search-modal-input-wrapper">
                <Search size={18} className="search-modal-icon" />
                <input
                  ref={modalInputRef}
                  type="text"
                  value={localQuery}
                  onChange={(e) => setLocalQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={isModelId(localQuery) ? "Finding similar models..." : "Fuzzy search models..."}
                  className="search-modal-input"
                  autoFocus
                />
                {localQuery.length > 0 && (
                  <button
                    className="search-modal-clear"
                    onClick={() => {
                      setLocalQuery('');
                      setResults([]);
                      modalInputRef.current?.focus();
                    }}
                    aria-label="Clear search"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
              <button className="search-modal-close" onClick={closeModal}>
                <X size={20} />
              </button>
            </div>

            <div className="search-modal-body" ref={resultsRef}>
              {isLoading && (
                <div className="search-modal-loading">
                  <div className="search-loading-spinner" />
                  <span>Searching...</span>
                </div>
              )}

              {!isLoading && localQuery.length >= 2 && results.length === 0 && (
                <div className="search-modal-empty">
                  <span>No models found for "{localQuery}"</span>
                  <p className="search-empty-hint">Try a different spelling or fewer characters</p>
                </div>
              )}

              {!isLoading && localQuery.length < 2 && (
                <div className="search-modal-hint">
                  <p>Start typing to search models</p>
                  <div className="search-hint-features">
                    <div className="search-hint-feature">
                      <Sparkles size={14} />
                      <span><strong>Fuzzy search</strong> — finds models even with typos</span>
                    </div>
                    <div className="search-hint-feature">
                      <Search size={14} />
                      <span><strong>Semantic search</strong> — enter a model ID to find similar models</span>
                    </div>
                  </div>
                  <div className="search-hint-examples">
                    <span>Try: <code>lama</code> (finds llama), <code>brt</code> (finds bert), <code>gpt2</code></span>
                  </div>
                </div>
              )}

              {!isLoading && results.length > 0 && (
                <>
                  <div className="search-results-header">
                    {searchType === 'semantic' ? (
                      <>Similar to "{localQuery}"</>
                    ) : (
                      <>{totalMatches.toLocaleString()} matches <span className="search-type-badge">fuzzy</span></>
                    )}
                  </div>
                  <div className="search-results-list">
                    {results.map((result, idx) => (
                      <div
                        key={result.model_id}
                        className={`search-result-item ${idx === selectedIndex ? 'selected' : ''}`}
                        onClick={() => handleSelect(result)}
                        onMouseEnter={() => setSelectedIndex(idx)}
                        role="option"
                        aria-selected={idx === selectedIndex}
                      >
                        <div className="search-result-content">
                          <div className="search-result-main">
                            <span className="search-result-id">{result.model_id}</span>
                            {result.similarity !== undefined && (
                              <span className="search-result-similarity" title="Semantic similarity">
                                {(result.similarity * 100).toFixed(1)}%
                              </span>
                            )}
                            {searchType === 'fuzzy' && result.score !== undefined && (
                              <span className="search-result-relevance" title="Match relevance">
                                {formatScore(result.score)}
                              </span>
                            )}
                          </div>
                          <div className="search-result-meta">
                            {result.library_name && (
                              <span className="search-result-tag">{result.library_name}</span>
                            )}
                            {result.pipeline_tag && (
                              <span className="search-result-tag">{result.pipeline_tag}</span>
                            )}
                            {result.downloads !== undefined && result.downloads > 0 && (
                              <span className="search-result-stat">
                                <Download size={10} />
                                {result.downloads >= 1000000 
                                  ? `${(result.downloads / 1000000).toFixed(1)}M`
                                  : result.downloads >= 1000
                                  ? `${(result.downloads / 1000).toFixed(0)}K`
                                  : result.downloads}
                              </span>
                            )}
                            {result.likes !== undefined && result.likes > 0 && (
                              <span className="search-result-stat">
                                <Heart size={10} />
                                {result.likes}
                              </span>
                            )}
                          </div>
                        </div>
                        <ArrowRight size={14} className="search-result-arrow" />
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="search-modal-footer">
              <div className="search-modal-shortcuts">
                <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
                <span><kbd>Enter</kbd> Select</span>
                <span><kbd>Esc</kbd> Close</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
