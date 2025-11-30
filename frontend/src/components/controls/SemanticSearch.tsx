/**
 * Enhanced semantic search component with depth and family filtering.
 * Supports natural language queries and semantic similarity search.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X } from 'lucide-react';
import { API_BASE } from '../../config/api';
import './SemanticSearch.css';

interface SemanticSearchResult {
  model_id: string;
  x: number;
  y: number;
  z: number;
  similarity?: number;
  family_depth?: number | null;
  parent_model?: string | null;
  downloads: number;
  likes: number;
  library_name?: string | null;
  pipeline_tag?: string | null;
}

interface SemanticSearchProps {
  onSelect?: (result: SemanticSearchResult) => void;
  onZoomTo?: (x: number, y: number, z: number) => void;
  onSearchComplete?: (results: SemanticSearchResult[]) => void;
}

export default function SemanticSearch({ 
  onSelect, 
  onZoomTo,
  onSearchComplete 
}: SemanticSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [searchMode, setSearchMode] = useState<'semantic' | 'text'>('semantic');
  const [depthFilter, setDepthFilter] = useState<number | null>(null);
  const [familyFilter, setFamilyFilter] = useState<string>('');
  const [queryModel, setQueryModel] = useState<string>('');
  
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Parse natural language query for depth and family hints
  const parseQuery = useCallback((q: string) => {
    const lower = q.toLowerCase();
    let depth: number | null = null;
    let family: string = '';
    let cleanQuery = q;

    // Extract depth hints: "depth 2", "at depth 3", "level 1", etc.
    const depthMatch = lower.match(/(?:depth|level|at depth)\s*(\d+)/);
    if (depthMatch) {
      depth = parseInt(depthMatch[1], 10);
      cleanQuery = cleanQuery.replace(new RegExp(depthMatch[0], 'gi'), '').trim();
    }

    // Extract family hints: "family Llama", "in Meta-Llama", etc.
    const familyMatch = lower.match(/(?:family|in|from)\s+([a-zA-Z0-9\-_\/]+)/);
    if (familyMatch) {
      family = familyMatch[1];
      cleanQuery = cleanQuery.replace(new RegExp(familyMatch[0], 'gi'), '').trim();
    }

    return { depth, family, cleanQuery };
  }, []);

  // Perform semantic search
  const performSemanticSearch = useCallback(async (queryModelId: string, depth: number | null, family: string) => {
    // Validate model ID format (should be org/model-name)
    if (!queryModelId || queryModelId.length < 3 || !queryModelId.includes('/')) {
      setResults([]);
      setIsOpen(false);
      setIsLoading(false);
      return;
    }
    
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        query_model_id: queryModelId,
        k: '100',
        min_downloads: '0',
        min_likes: '0',
        projection_method: 'umap',
      });

      const response = await fetch(`${API_BASE}/api/models/semantic-similarity?${params}`);
      
      if (response.status === 404) {
        // Model not found - this is expected for some models
        setResults([]);
        setIsOpen(false);
        setIsLoading(false);
        return;
      }
      
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = 'Semantic search failed';
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || errorMessage;
        } catch {
          errorMessage = `Error ${response.status}: ${errorText || errorMessage}`;
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      let models = data.models || [];

      // Filter by depth if specified
      if (depth !== null) {
        models = models.filter((m: SemanticSearchResult) => 
          m.family_depth !== null && m.family_depth !== undefined && m.family_depth === depth
        );
      }

      // Filter by family if specified
      if (family) {
        models = models.filter((m: SemanticSearchResult) => {
          const modelIdLower = m.model_id.toLowerCase();
          const parentLower = (m.parent_model || '').toLowerCase();
          const familyLower = family.toLowerCase();
          return modelIdLower.includes(familyLower) || parentLower.includes(familyLower);
        });
      }

      // Sort by similarity (highest first)
      models.sort((a: SemanticSearchResult, b: SemanticSearchResult) => 
        (b.similarity || 0) - (a.similarity || 0)
      );

      setResults(models.slice(0, 20)); // Top 20
      setIsOpen(true);
      setSelectedIndex(-1);
      
      if (onSearchComplete) {
        onSearchComplete(models);
      }
    } catch {
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [onSearchComplete]);

  // Perform text search with depth/family filters
  const performTextSearch = useCallback(async (searchQuery: string, depth: number | null, family: string) => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        limit: '50',
      });

      const response = await fetch(`${API_BASE}/api/search?${params}`);
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      let models = data.results || [];

      // Filter by depth if specified (would need to fetch family tree data)
      // For now, we'll filter by family name in model_id
      if (family) {
        models = models.filter((m: any) => {
          const modelIdLower = m.model_id.toLowerCase();
          const parentLower = (m.parent_model || '').toLowerCase();
          const familyLower = family.toLowerCase();
          return modelIdLower.includes(familyLower) || parentLower.includes(familyLower);
        });
      }

      // Convert to SemanticSearchResult format
      const formattedResults: SemanticSearchResult[] = models.map((m: any) => ({
        model_id: m.model_id,
        x: m.x || 0,
        y: m.y || 0,
        z: m.z || 0,
        downloads: m.downloads || 0,
        likes: m.likes || 0,
        library_name: m.library,
        pipeline_tag: m.pipeline,
        parent_model: m.parent_model,
      }));

      setResults(formattedResults.slice(0, 20));
      setIsOpen(true);
      setSelectedIndex(-1);
      
      if (onSearchComplete) {
        onSearchComplete(formattedResults);
      }
    } catch {
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [onSearchComplete]);

  // Handle search
  useEffect(() => {
    if (query.length < 2 && !queryModel) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const timer = setTimeout(() => {
      const { depth, family, cleanQuery } = parseQuery(query);
      
      // Update filters from parsed query
      if (depth !== null) setDepthFilter(depth);
      if (family) setFamilyFilter(family);

      if (searchMode === 'semantic' && queryModel) {
        performSemanticSearch(queryModel, depth || depthFilter, family || familyFilter);
      } else if (cleanQuery.length >= 2) {
        performTextSearch(cleanQuery, depth || depthFilter, family || familyFilter);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query, queryModel, searchMode, depthFilter, familyFilter, parseQuery, performSemanticSearch, performTextSearch]);

  const handleSelect = useCallback((result: SemanticSearchResult) => {
    if (onZoomTo && result.x !== undefined && result.y !== undefined) {
      onZoomTo(result.x, result.y, result.z || 0);
    }
    
    if (onSelect) {
      onSelect(result);
    }
    
    setIsOpen(false);
    setQuery('');
    setQueryModel('');
    inputRef.current?.blur();
  }, [onSelect, onZoomTo]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen || results.length === 0) return;
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < results.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && results[selectedIndex]) {
        handleSelect(results[selectedIndex]);
      } else if (results.length > 0) {
        handleSelect(results[0]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
      inputRef.current?.blur();
    }
  };

  return (
    <div className="semantic-search-container">
      <div className="semantic-search-controls">
        <div className="search-mode-toggle">
          <button
            className={`mode-btn ${searchMode === 'semantic' ? 'active' : ''}`}
            onClick={() => setSearchMode('semantic')}
            title="Semantic similarity search"
          >
            Semantic
          </button>
          <button
            className={`mode-btn ${searchMode === 'text' ? 'active' : ''}`}
            onClick={() => setSearchMode('text')}
            title="Text search"
          >
            Text
          </button>
        </div>

        {searchMode === 'semantic' && (
          <input
            type="text"
            value={queryModel}
            onChange={(e) => setQueryModel(e.target.value)}
            placeholder="Reference model ID (e.g., Meta-Llama-3.1-8B-Instruct)"
            className="query-model-input"
          />
        )}

        <div className="search-filters">
          <input
            type="number"
            value={depthFilter || ''}
            onChange={(e) => setDepthFilter(e.target.value ? parseInt(e.target.value, 10) : null)}
            placeholder="Depth"
            min="0"
            className="depth-filter-input"
            title="Filter by family depth (0 = root)"
          />
          <input
            type="text"
            value={familyFilter}
            onChange={(e) => setFamilyFilter(e.target.value)}
            placeholder="Family name"
            className="family-filter-input"
            title="Filter by family name"
          />
        </div>
      </div>

      <div className="semantic-search-bar">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setIsOpen(true)}
          placeholder={
            searchMode === 'semantic'
              ? "e.g., 'depth 2 Llama models' or 'family Meta-Llama depth 3'"
              : "Search models, orgs, tasks..."
          }
          className="semantic-search-input"
          aria-label="Semantic search"
        />
        {isLoading && <div className="search-loading">Loading...</div>}
        {query.length > 0 && !isLoading && (
          <button
            className="search-clear"
            onClick={() => {
              setQuery('');
              setResults([]);
              setIsOpen(false);
            }}
            aria-label="Clear search"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div ref={resultsRef} className="semantic-search-results" role="listbox">
          {results.map((result, idx) => (
            <div
              key={result.model_id}
              className={`semantic-search-result ${idx === selectedIndex ? 'selected' : ''}`}
              onClick={() => handleSelect(result)}
              role="option"
              aria-selected={idx === selectedIndex}
            >
              <div className="result-header">
                <strong className="result-model-id">{result.model_id}</strong>
                {result.similarity !== undefined && (
                  <span className="similarity-badge">
                    {(result.similarity * 100).toFixed(1)}% similar
                  </span>
                )}
              </div>
              <div className="result-meta">
                {result.family_depth !== null && result.family_depth !== undefined && (
                  <span className="result-tag depth-tag">Depth {result.family_depth}</span>
                )}
                {result.library_name && (
                  <span className="result-tag">{result.library_name}</span>
                )}
                {result.pipeline_tag && (
                  <span className="result-tag">{result.pipeline_tag}</span>
                )}
                <span className="result-stats">
                  {result.downloads.toLocaleString()} downloads • {result.likes.toLocaleString()} likes
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {isOpen && query.length >= 2 && results.length === 0 && !isLoading && (
        <div className="semantic-search-results">
          <div className="search-no-results">No results found</div>
        </div>
      )}
    </div>
  );
}

