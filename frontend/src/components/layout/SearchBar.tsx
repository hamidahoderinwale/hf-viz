/**
 * Enhanced search bar with autocomplete and keyboard navigation.
 * Integrates with filter store and triggers map zoom/modal open.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useFilterStore } from '../../stores/filterStore';
import './SearchBar.css';

import { API_BASE } from '../../config/api';

interface SearchResult {
  model_id: string;
  x: number;
  y: number;
  z: number;
  org: string;
  library?: string;
  pipeline?: string;
  license?: string;
  snippet?: string;
  match_score?: number;
}

interface SearchBarProps {
  onSelect?: (result: SearchResult) => void;
  onZoomTo?: (x: number, y: number, z: number) => void;
}

export default function SearchBar({ onSelect, onZoomTo }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  
  const setSearchQuery = useFilterStore((state) => state.setSearchQuery);
  
  // Debounced search
  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }
    
    setIsLoading(true);
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(
          `${API_BASE}/api/search?q=${encodeURIComponent(query)}&limit=20`
        );
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        setResults(data.results || []);
        setIsOpen(true);
        setSelectedIndex(-1);
      } catch (err) {
        console.error('Search error:', err);
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 150);
    
    return () => clearTimeout(timer);
  }, [query]);
  
  const handleSelect = useCallback((result: SearchResult) => {
    setSearchQuery(result.model_id);
    
    // Trigger zoom if coordinates available
    if (onZoomTo && result.x !== undefined && result.y !== undefined) {
      onZoomTo(result.x, result.y, result.z || 0);
    }
    
    // Trigger select callback
    if (onSelect) {
      onSelect(result);
    }
    
    setIsOpen(false);
    setQuery('');
    inputRef.current?.blur();
  }, [onSelect, onZoomTo, setSearchQuery]);
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen || results.length === 0) return;
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < results.length - 1 ? prev + 1 : prev
      );
      // Scroll into view
      if (resultsRef.current && selectedIndex >= 0) {
        const selectedElement = resultsRef.current.children[selectedIndex + 1] as HTMLElement;
        selectedElement?.scrollIntoView({ block: 'nearest' });
      }
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
  
  const handleFocus = () => {
    if (results.length > 0) {
      setIsOpen(true);
    }
  };
  
  const handleBlur = (e: React.FocusEvent) => {
    // Delay to allow click events on results
    setTimeout(() => {
      if (!resultsRef.current?.contains(document.activeElement)) {
        setIsOpen(false);
      }
    }, 200);
  };
  
  return (
    <div className="search-bar-container">
      <div className="search-bar">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder="Search models, orgs, tasks, licenses..."
          className="search-input"
          aria-label="Search models"
          aria-expanded={isOpen}
          aria-haspopup="listbox"
        />
        {isLoading && <div className="search-loading">⟳</div>}
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
            ×
          </button>
        )}
      </div>
      {isOpen && results.length > 0 && (
        <div ref={resultsRef} className="search-results" role="listbox">
          {results.map((result, idx) => (
            <div
              key={result.model_id}
              className={`search-result ${idx === selectedIndex ? 'selected' : ''}`}
              onClick={() => handleSelect(result)}
              role="option"
              aria-selected={idx === selectedIndex}
            >
              <div className="result-header">
                <strong className="result-model-id">{result.model_id}</strong>
                {result.org && <span className="result-org">{result.org}</span>}
              </div>
              <div className="result-meta">
                {result.library && <span className="result-tag">{result.library}</span>}
                {result.pipeline && <span className="result-tag">{result.pipeline}</span>}
                {result.license && <span className="result-tag">{result.license}</span>}
              </div>
              {result.snippet && (
                <div 
                  className="result-snippet" 
                  dangerouslySetInnerHTML={{ __html: result.snippet }} 
                />
              )}
            </div>
          ))}
        </div>
      )}
      {isOpen && query.length >= 2 && results.length === 0 && !isLoading && (
        <div className="search-results">
          <div className="search-no-results">No results found</div>
        </div>
      )}
    </div>
  );
}

