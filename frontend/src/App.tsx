/**
 * Main React app component using Visx for visualization.
 */
import React, { useState, useEffect, useCallback, useRef, useMemo, lazy, Suspense } from 'react';
import EnhancedScatterPlot from './components/EnhancedScatterPlot';
import NetworkGraph from './components/NetworkGraph';
import Histogram from './components/Histogram';
import UVProjectionSquare from './components/UVProjectionSquare';
import ModelModal from './components/ModelModal';
import PaperPlots from './components/PaperPlots';
import LiveModelCount from './components/LiveModelCount';
import ColorLegend from './components/ColorLegend';
import ModelTooltip from './components/ModelTooltip';
import { ModelPoint, Stats, FamilyTree, SearchResult, SimilarModel } from './types';
import cache, { IndexedDBCache } from './utils/indexedDB';
import { debounce } from './utils/debounce';
import requestManager from './utils/requestManager';
import './App.css';

const ScatterPlot3D = lazy(() => import('./components/ScatterPlot3D'));

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [data, setData] = useState<ModelPoint[]>([]);
  const [filteredCount, setFilteredCount] = useState<number | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelPoint | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModels, setSelectedModels] = useState<ModelPoint[]>([]);
  const [viewMode, setViewMode] = useState<'scatter' | 'network' | 'histogram' | '3d' | 'paper-plots'>('3d');
  const [histogramAttribute, setHistogramAttribute] = useState<'downloads' | 'likes' | 'trending_score'>('downloads');
  const [baseModelsOnly, setBaseModelsOnly] = useState(false);
  const [semanticSimilarityMode, setSemanticSimilarityMode] = useState(false);
  const [semanticQueryModel, setSemanticQueryModel] = useState<string | null>(null);
  
  const [familyTree, setFamilyTree] = useState<ModelPoint[]>([]);
  const [familyTreeModelId, setFamilyTreeModelId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchInput, setSearchInput] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [viewCenter, setViewCenter] = useState<{ x: number; y: number; z: number } | null>(null);
  const [projectionMethod, setProjectionMethod] = useState<'umap' | 'tsne'>('umap');
  const [bookmarkedModels, setBookmarkedModels] = useState<string[]>([]);
  const [comparisonModels, setComparisonModels] = useState<ModelPoint[]>([]);
  const [similarModels, setSimilarModels] = useState<SimilarModel[]>([]);
  const [showSimilar, setShowSimilar] = useState(false);

  const [minDownloads, setMinDownloads] = useState(0);
  const [minLikes, setMinLikes] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [colorBy, setColorBy] = useState('library_name');
  const [sizeBy, setSizeBy] = useState('downloads');
  const [colorScheme, setColorScheme] = useState<'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm'>('viridis');
  const [showLegend, setShowLegend] = useState(true);
  const [hoveredModel, setHoveredModel] = useState<ModelPoint | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  
  const activeFilterCount = (minDownloads > 0 ? 1 : 0) + 
                           (minLikes > 0 ? 1 : 0) + 
                           (searchQuery.length > 0 ? 1 : 0);
  
  const resetFilters = useCallback(() => {
    setMinDownloads(0);
    setMinLikes(0);
    setSearchQuery('');
  }, []);

  const [width, setWidth] = useState(window.innerWidth * 0.7);
  const [height, setHeight] = useState(window.innerHeight * 0.7);

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth * 0.7);
      setHeight(window.innerHeight * 0.7);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const fetchDataAbortRef = useRef<(() => void) | null>(null);

  const fetchData = useCallback(async () => {
    if (fetchDataAbortRef.current) {
      fetchDataAbortRef.current();
    }

    setLoading(true);
    setError(null);
    
    try {
      const cacheKey = IndexedDBCache.generateCacheKey({
        minDownloads,
        minLikes,
        searchQuery,
        projectionMethod,
      });

      const cachedModels = await cache.getCachedModels(cacheKey);
      if (cachedModels) {
        setData(cachedModels);
        setLoading(false);
        return;
      }
      let models: ModelPoint[];
      let count: number | null = null;
      
      if (semanticSimilarityMode && semanticQueryModel) {
        const params = new URLSearchParams({
          query_model_id: semanticQueryModel,
          k: '500',
          min_downloads: minDownloads.toString(),
          min_likes: minLikes.toString(),
          projection_method: projectionMethod,
        });
        const url = `${API_BASE}/api/models/semantic-similarity?${params}`;
        const response = await requestManager.fetch(url, {}, cacheKey);
        if (!response.ok) throw new Error('Failed to fetch similar models');
        const result = await response.json();
        models = result.models || [];
        // Semantic similarity doesn't return filtered_count, use models length
        count = models.length;
      } else {
        const params = new URLSearchParams({
          min_downloads: minDownloads.toString(),
          min_likes: minLikes.toString(),
          color_by: colorBy,
          size_by: sizeBy,
          projection_method: projectionMethod,
          base_models_only: baseModelsOnly.toString(),
        });
        if (searchQuery) {
          params.append('search_query', searchQuery);
        }

        const url = `${API_BASE}/api/models?${params}`;
        const response = await requestManager.fetch(url, {}, cacheKey);
        if (!response.ok) throw new Error('Failed to fetch models');
        const result = await response.json();
        
        // Handle both old format (array) and new format (object with models, filtered_count, returned_count)
        if (Array.isArray(result)) {
          models = result;
          count = models.length;
        } else {
          models = result.models || [];
          count = result.filtered_count ?? models.length;
        }
      }
      
      await cache.cacheModels(cacheKey, models);
      
      setData(models);
      setFilteredCount(count);
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    } finally {
      setLoading(false);
      fetchDataAbortRef.current = null;
    }
  }, [minDownloads, minLikes, searchQuery, colorBy, sizeBy, projectionMethod, baseModelsOnly, semanticSimilarityMode, semanticQueryModel]);

  const debouncedFetchData = useMemo(
    () => debounce(fetchData, 300),
    [fetchData]
  );

  useEffect(() => {
    if (searchQuery) {
      debouncedFetchData();
    } else {
      // Immediate fetch if search is cleared
      fetchData();
    }
    return () => {
      debouncedFetchData.cancel();
    };
  }, [searchQuery, debouncedFetchData, fetchData]);

  useEffect(() => {
    debouncedFetchData();
    return () => {
      debouncedFetchData.cancel();
    };
  }, [minDownloads, minLikes, colorBy, sizeBy, baseModelsOnly, projectionMethod, semanticSimilarityMode, semanticQueryModel, debouncedFetchData]);

  useEffect(() => {
    const fetchStats = async () => {
      const cacheKey = 'stats';
      const cachedStats = await cache.getCachedStats(cacheKey);
      if (cachedStats) {
        setStats(cachedStats);
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        const statsData = await response.json();
        await cache.cacheStats(cacheKey, statsData);
        setStats(statsData);
      } catch (err) {
        console.error('Error fetching stats:', err);
      }
    };

    fetchStats();
  }, []);

  // Search models for family tree lookup
  const searchModels = useCallback(async (query: string) => {
    if (query.length < 1) {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/search?query=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setSearchResults(data.results || []);
      setShowSearchResults(true);
    } catch (err) {
      console.error('Search error:', err);
      setSearchResults([]);
    }
  }, []);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      searchModels(searchInput);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput, searchModels]);

  const loadFamilyTree = useCallback(async (modelId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/family/${encodeURIComponent(modelId)}?max_depth=5`);
      if (!response.ok) throw new Error('Failed to load family tree');
      const data: FamilyTree = await response.json();
      setFamilyTree(data.family || []);
      setFamilyTreeModelId(modelId);
      setShowSearchResults(false);
      setSearchInput('');
    } catch (err) {
      console.error('Family tree error:', err);
      setFamilyTree([]);
      setFamilyTreeModelId(null);
    }
  }, []);

  const clearFamilyTree = useCallback(() => {
    setFamilyTree([]);
    setFamilyTreeModelId(null);
  }, []);

  const loadSimilarModels = useCallback(async (modelId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/similar/${encodeURIComponent(modelId)}?k=10`);
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = 'Failed to load similar models';
        if (response.status === 404) {
          errorMessage = 'Model not found';
        } else if (response.status === 503) {
          errorMessage = 'Data not loaded yet. Please wait a moment and try again.';
        } else {
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.detail || errorMessage;
          } catch {
            errorMessage = `Error ${response.status}: ${errorText || errorMessage}`;
          }
        }
        throw new Error(errorMessage);
      }
      const data = await response.json();
      setSimilarModels(data.similar_models || []);
      setShowSimilar(true);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load similar models';
      console.error('Similar models error:', errorMessage, err);
      // Only show error if it's not a silent failure (e.g., user cancelled)
      if (errorMessage !== 'Failed to load similar models' || !(err instanceof TypeError && err.message.includes('fetch'))) {
        setError(`Similar models: ${errorMessage}`);
        // Clear error after 5 seconds
        setTimeout(() => setError(null), 5000);
      }
      setSimilarModels([]);
      setShowSimilar(false);
    }
  }, []);

  // Bookmark management
  const toggleBookmark = useCallback((modelId: string) => {
    setBookmarkedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  }, []);

  // Comparison management
  const addToComparison = useCallback((model: ModelPoint) => {
    if (comparisonModels.length < 3 && !comparisonModels.find(m => m.model_id === model.model_id)) {
      setComparisonModels(prev => [...prev, model]);
    }
  }, [comparisonModels]);

  const removeFromComparison = useCallback((modelId: string) => {
    setComparisonModels(prev => prev.filter(m => m.model_id !== modelId));
  }, []);

  // Export selected models
  const exportModels = useCallback(async (modelIds: string[]) => {
    try {
      const response = await fetch(`${API_BASE}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelIds),
      });
      if (!response.ok) throw new Error('Export failed');
      const data = await response.json();
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `models_export_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export models');
    }
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face</h1>
        <p style={{ maxWidth: '900px', margin: '0 auto', lineHeight: '1.6' }}>
          Many have observed that the development and deployment of generative machine learning (ML) and artificial intelligence (AI) models follow a distinctive pattern in which pre-trained models are adapted and fine-tuned for specific downstream tasks. However, there is limited empirical work that examines the structure of these interactions. This paper analyzes 1.86 million models on Hugging Face, a leading peer production platform for model development. Our study of model family trees reveals sprawling fine-tuning lineages that vary widely in size and structure. Using an evolutionary biology lens, we measure genetic similarity and mutation of traits over model families.
          {' '}
          <a 
            href="https://arxiv.org/abs/2508.06811" 
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'white', textDecoration: 'underline', fontWeight: '500' }}
          >
            Read the full paper
          </a>
        </p>
        <p style={{ marginTop: '1rem', fontSize: '0.9rem', opacity: 0.9, lineHeight: '1.6' }}>
          <strong>Resources:</strong>{' '}
          <a 
            href="https://github.com/bendlaufer/ai-ecosystem" 
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#64b5f6', textDecoration: 'underline', marginRight: '1rem' }}
          >
            GitHub Repository
          </a>
          <a 
            href="https://huggingface.co/modelbiome" 
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#64b5f6', textDecoration: 'underline' }}
          >
            Hugging Face Dataset
          </a>
        </p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>
          <strong>Authors:</strong> Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center', width: '100%' }}>
          <LiveModelCount compact={true} />
          {stats && (
            <div className="stats">
              <span>Dataset Models: {stats.total_models.toLocaleString()}</span>
              <span>Libraries: {stats.unique_libraries}</span>
              <span>Task Types: {stats.unique_task_types ?? stats.unique_pipelines}</span>
            </div>
          )}
        </div>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          {/* Live Model Count - Prominent Display */}
          <LiveModelCount compact={false} />
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', marginTop: '1rem' }}>
            <h2 style={{ margin: 0 }}>Filters</h2>
            {activeFilterCount > 0 && (
              <div style={{ 
                fontSize: '0.75rem', 
                background: '#4a4a4a', 
                color: 'white', 
                padding: '0.25rem 0.5rem', 
                borderRadius: '12px',
                fontWeight: '500'
              }}>
                {activeFilterCount} active
              </div>
            )}
          </div>
          
          {/* Filter Results Count */}
          {!loading && data.length > 0 && (
            <div className="sidebar-section" style={{ 
              background: '#e8f5e9', 
              borderColor: '#c8e6c9',
              fontSize: '0.9rem'
            }}>
              <strong>{data.length.toLocaleString()}</strong> {data.length === 1 ? 'model' : 'models'} shown
              {filteredCount !== null && filteredCount !== data.length && (
                <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.25rem' }}>
                  of {filteredCount.toLocaleString()} matching filters
                </div>
              )}
              {stats && filteredCount !== null && (
                <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.25rem' }}>
                  {filteredCount < stats.total_models && (
                    <>out of {stats.total_models.toLocaleString()} total in dataset</>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Search Section */}
          <div className="sidebar-section">
            <label style={{ fontWeight: '600', marginBottom: '0.5rem', display: 'block' }}>
              Search Models
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by model ID, tags, or keywords..."
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
              Searches model names, tags, and metadata
            </div>
          </div>

          {/* Popularity Filters */}
          <div className="sidebar-section">
            <h3>Popularity Filters</h3>
            
            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: '500' }}>Minimum Downloads</span>
                <span style={{ fontWeight: '600', color: '#4a4a4a' }}>
                  {minDownloads > 0 ? minDownloads.toLocaleString() : 'Any'}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max={stats ? Math.min(1000000, Math.ceil(stats.avg_downloads * 10)) : 1000000}
                step={stats && stats.avg_downloads > 1000 ? "1000" : "100"}
                value={minDownloads}
                onChange={(e) => setMinDownloads(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginTop: '0.25rem' }}>
                <span>0</span>
                <span>{stats ? Math.ceil(stats.avg_downloads).toLocaleString() : '100K'} avg</span>
                <span>{stats ? Math.min(1000000, Math.ceil(stats.avg_downloads * 10)).toLocaleString() : '1M'}</span>
              </div>
            </label>

            <label style={{ display: 'block' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: '500' }}>Minimum Likes</span>
                <span style={{ fontWeight: '600', color: '#4a4a4a' }}>
                  {minLikes > 0 ? minLikes.toLocaleString() : 'Any'}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max={stats ? Math.min(10000, Math.ceil(stats.avg_likes * 20)) : 10000}
                step={stats && stats.avg_likes > 10 ? "10" : "1"}
                value={minLikes}
                onChange={(e) => setMinLikes(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginTop: '0.25rem' }}>
                <span>0</span>
                <span>{stats ? Math.ceil(stats.avg_likes).toLocaleString() : '10'} avg</span>
                <span>{stats ? Math.min(10000, Math.ceil(stats.avg_likes * 20)).toLocaleString() : '10K'}</span>
              </div>
            </label>
          </div>

          {/* Visualization Options */}
          <div className="sidebar-section">
            <h3>Visualization Options</h3>
            
            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>View Mode</span>
              <select 
                value={viewMode} 
                onChange={(e) => setViewMode(e.target.value as any)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
              >
                <option value="3d">3D Latent Space</option>
                <option value="scatter">2D Latent Space</option>
                <option value="network">Network Graph</option>
                <option value="histogram">Distribution Histogram</option>
                <option value="paper-plots">Paper Visualizations</option>
              </select>
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                {viewMode === '3d' && 'Interactive 3D exploration of model relationships'}
                {viewMode === 'scatter' && '2D projection showing model similarity'}
                {viewMode === 'network' && 'Network graph of model connections'}
                {viewMode === 'histogram' && 'Distribution analysis of model attributes'}
                {viewMode === 'paper-plots' && 'Interactive visualizations from the research paper'}
              </div>
            </label>

            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>Color Encoding</span>
              <select 
                value={colorBy} 
                onChange={(e) => setColorBy(e.target.value)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
              >
                <option value="library_name">Library (e.g., transformers, diffusers)</option>
                <option value="pipeline_tag">Pipeline/Task Type</option>
                <option value="cluster_id">Cluster (semantic groups)</option>
                <option value="family_depth">Family Tree Depth</option>
                <option value="downloads">Download Count</option>
                <option value="likes">Like Count</option>
                <option value="trending_score">Trending Score</option>
                <option value="licenses">License Type</option>
              </select>
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                {colorBy === 'cluster_id' && 'Semantic clusters from embeddings'}
                {colorBy === 'family_depth' && 'Generation depth in family tree'}
                {colorBy === 'licenses' && 'Model license types'}
              </div>
            </label>

            {/* Color Scheme Selector (for continuous scales) */}
            {(colorBy === 'downloads' || colorBy === 'likes' || colorBy === 'family_depth' || colorBy === 'trending_score') && (
              <label style={{ marginBottom: '1rem', display: 'block' }}>
                <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>Color Scheme</span>
                <select 
                  value={colorScheme} 
                  onChange={(e) => setColorScheme(e.target.value as any)}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
                >
                  <option value="viridis">Viridis (blue to yellow)</option>
                  <option value="plasma">Plasma (purple to yellow)</option>
                  <option value="inferno">Inferno (black to yellow)</option>
                  <option value="magma">Magma (black to pink)</option>
                  <option value="coolwarm">Cool-Warm (blue to red)</option>
                </select>
              </label>
            )}

            {/* Legend Toggle */}
            <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                checked={showLegend}
                onChange={(e) => setShowLegend(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
              <span style={{ fontWeight: '500' }}>Show Color Legend</span>
            </label>

            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>Size Encoding</span>
              <select 
                value={sizeBy} 
                onChange={(e) => setSizeBy(e.target.value)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
              >
                <option value="downloads">Downloads (larger = more popular)</option>
                <option value="likes">Likes (larger = more liked)</option>
                <option value="trendingScore">Trending Score</option>
                <option value="none">Uniform Size</option>
              </select>
            </label>

            <div className="sidebar-section" style={{ background: '#fff3cd', borderColor: '#ffc107', marginBottom: '1rem', padding: '0.75rem', borderRadius: '4px', border: '1px solid' }}>
              <label style={{ display: 'block', marginBottom: '0' }}>
                <span style={{ fontWeight: '600', display: 'block', marginBottom: '0.5rem', color: '#856404' }}>
                  ⚙️ Projection Method
                </span>
                <select 
                  value={projectionMethod} 
                  onChange={(e) => setProjectionMethod(e.target.value as 'umap' | 'tsne')}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0', fontWeight: '500' }}
                >
                  <option value="umap">UMAP (better global structure)</option>
                  <option value="tsne">t-SNE (better local clusters)</option>
                </select>
                <div style={{ fontSize: '0.75rem', color: '#856404', marginTop: '0.5rem', lineHeight: '1.4' }}>
                  <strong>UMAP:</strong> Preserves global structure, better for exploring relationships<br/>
                  <strong>t-SNE:</strong> Emphasizes local clusters, better for finding groups
                </div>
              </label>
            </div>
          </div>

          {/* View Modes */}
          <div className="sidebar-section" style={{ background: '#f0f7ff', borderColor: '#b3d9ff' }}>
            <h3>View Modes</h3>
            
            <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={baseModelsOnly}
                onChange={(e) => setBaseModelsOnly(e.target.checked)}
                style={{ marginRight: '0.5rem', cursor: 'pointer' }}
              />
              <div>
                <span style={{ fontWeight: '500' }}>Base Models Only</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Show only root models (no parent). Click any model to see its family tree.
                </div>
              </div>
            </label>

            <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={semanticSimilarityMode}
                onChange={(e) => {
                  setSemanticSimilarityMode(e.target.checked);
                  if (!e.target.checked) {
                    setSemanticQueryModel(null);
                  }
                }}
                style={{ marginRight: '0.5rem', cursor: 'pointer' }}
              />
              <div>
                <span style={{ fontWeight: '500' }}>Semantic Similarity View</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Show models sorted by semantic similarity to a query model
                </div>
              </div>
            </label>

            {semanticSimilarityMode && (
              <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'white', borderRadius: '4px', border: '1px solid #d0d0d0' }}>
                <label style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>
                  Query Model ID
                </label>
                <input
                  type="text"
                  value={semanticQueryModel || ''}
                  onChange={(e) => setSemanticQueryModel(e.target.value || null)}
                  placeholder="e.g., bert-base-uncased"
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
                />
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem' }}>
                  {selectedModel && (
                    <button
                      onClick={() => setSemanticQueryModel(selectedModel.model_id)}
                      style={{
                        padding: '0.25rem 0.5rem',
                        background: '#4a90e2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem'
                      }}
                    >
                      Use Selected Model
                    </button>
                  )}
                  <div style={{ marginTop: '0.5rem' }}>
                    Enter a model ID or click a model in the visualization, then click "Use Selected Model"
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Quick Filters */}
          <div className="sidebar-section">
            <h3>Quick Filters</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              <button
                onClick={() => {
                  setMinDownloads(stats ? Math.ceil(stats.avg_downloads) : 100);
                  setMinLikes(0);
                }}
                className="btn btn-small"
                style={{
                  background: '#e3f2fd',
                  color: '#1976d2',
                  border: '1px solid #90caf9'
                }}
              >
                Above Avg Downloads
              </button>
              <button
                onClick={() => {
                  setMinDownloads(10000);
                  setMinLikes(10);
                }}
                className="btn btn-small"
                style={{
                  background: '#fff3e0',
                  color: '#e65100',
                  border: '1px solid #ffb74d'
                }}
              >
                Popular Models
              </button>
              <button
                onClick={resetFilters}
                disabled={activeFilterCount === 0}
                className="btn btn-small btn-secondary"
                style={{
                  opacity: activeFilterCount > 0 ? 1 : 0.5,
                  cursor: activeFilterCount > 0 ? 'pointer' : 'not-allowed'
                }}
              >
                Reset All
              </button>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>Family Tree Explorer</h3>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onFocus={() => searchInput.length > 0 && setShowSearchResults(true)}
                placeholder="Type model name..."
                style={{ width: '100%', padding: '0.5rem', borderRadius: '2px', border: '1px solid #d0d0d0' }}
              />
              {showSearchResults && searchResults.length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  background: 'white',
                  border: '1px solid #d0d0d0',
                  borderRadius: '2px',
                  marginTop: '2px',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  zIndex: 1000,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  {searchResults.map((result) => (
                    <div
                      key={result.model_id}
                      onClick={() => {
                        loadFamilyTree(result.model_id);
                      }}
                      style={{
                        padding: '0.5rem',
                        cursor: 'pointer',
                        borderBottom: '1px solid #f0f0f0',
                        fontSize: '0.85rem'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = '#f5f5f5'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'white'}
                    >
                      <div style={{ fontWeight: '500' }}>{result.model_id}</div>
                      {result.library_name && (
                        <div style={{ fontSize: '0.75rem', color: '#666' }}>{result.library_name}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            {familyTreeModelId && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                <div style={{ marginBottom: '0.25rem' }}>
                  <strong>Showing family tree for:</strong> {familyTreeModelId}
                </div>
                <div style={{ marginBottom: '0.5rem', color: '#666' }}>
                  {familyTree.length} family members
                </div>
                <button
                  onClick={clearFamilyTree}
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.8rem',
                    background: '#6a6a6a',
                    color: 'white',
                    border: 'none',
                    borderRadius: '2px',
                    cursor: 'pointer'
                  }}
                >
                  Clear Family Tree
                </button>
              </div>
            )}
          </div>

          {/* Bookmarks */}
          {bookmarkedModels.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '2px', border: '1px solid #d0d0d0' }}>
              <h3 style={{ marginTop: 0, fontSize: '0.9rem', fontWeight: '600' }}>Bookmarks ({bookmarkedModels.length})</h3>
              <div style={{ maxHeight: '150px', overflowY: 'auto', fontSize: '0.85rem' }}>
                {bookmarkedModels.map(modelId => (
                  <div key={modelId} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{modelId}</span>
                    <button
                      onClick={() => toggleBookmark(modelId)}
                      style={{ marginLeft: '0.5rem', padding: '0.1rem 0.3rem', fontSize: '0.7rem', background: '#6a6a6a', color: 'white', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
              <button
                onClick={() => exportModels(bookmarkedModels)}
                style={{ marginTop: '0.5rem', padding: '0.25rem 0.5rem', fontSize: '0.8rem', background: '#4a4a4a', color: 'white', border: 'none', borderRadius: '2px', cursor: 'pointer', width: '100%' }}
              >
                Export Bookmarks
              </button>
            </div>
          )}

          {/* Comparison */}
          {comparisonModels.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '2px', border: '1px solid #d0d0d0' }}>
              <h3 style={{ marginTop: 0, fontSize: '0.9rem', fontWeight: '600' }}>Comparison ({comparisonModels.length}/3)</h3>
              {comparisonModels.map(model => (
                <div key={model.model_id} style={{ marginBottom: '0.5rem', padding: '0.5rem', background: 'white', borderRadius: '2px', fontSize: '0.85rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong>{model.model_id}</strong>
                    <button
                      onClick={() => removeFromComparison(model.model_id)}
                      style={{ padding: '0.1rem 0.3rem', fontSize: '0.7rem', background: '#6a6a6a', color: 'white', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                    >
                      Remove
                    </button>
                  </div>
                  <div style={{ marginTop: '0.25rem', fontSize: '0.75rem', color: '#666' }}>
                    {model.library_name && <span>Library: {model.library_name} | </span>}
                    Downloads: {model.downloads.toLocaleString()} | Likes: {model.likes.toLocaleString()}
                  </div>
                </div>
              ))}
              <button
                onClick={() => setComparisonModels([])}
                style={{ marginTop: '0.5rem', padding: '0.25rem 0.5rem', fontSize: '0.8rem', background: '#6a6a6a', color: 'white', border: 'none', borderRadius: '2px', cursor: 'pointer', width: '100%' }}
              >
                Clear Comparison
              </button>
            </div>
          )}

          {/* Similar Models */}
          {showSimilar && similarModels.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '2px', border: '1px solid #d0d0d0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h3 style={{ marginTop: 0, fontSize: '0.9rem', fontWeight: '600' }}>Similar Models</h3>
                <button
                  onClick={() => setShowSimilar(false)}
                  style={{ padding: '0.1rem 0.3rem', fontSize: '0.7rem', background: '#6a6a6a', color: 'white', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                >
                  Close
                </button>
              </div>
              <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.85rem' }}>
                {similarModels.map((similar, idx) => (
                  <div key={idx} style={{ marginBottom: '0.5rem', padding: '0.5rem', background: 'white', borderRadius: '2px' }}>
                    <div style={{ fontWeight: '500' }}>{similar.model_id}</div>
                    <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                      Similarity: {(similar.similarity * 100).toFixed(1)}% | Distance: {similar.distance.toFixed(3)}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#666' }}>
                      {similar.library_name && <span>{similar.library_name} | </span>}
                      Downloads: {similar.downloads.toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {viewMode === 'histogram' && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: '#f9f9f9', borderRadius: '4px', border: '1px solid #e0e0e0' }}>
              <label style={{ display: 'block' }}>
                <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>Histogram Attribute</span>
                <select 
                  value={histogramAttribute} 
                  onChange={(e) => setHistogramAttribute(e.target.value as any)}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
                >
                  <option value="downloads">Downloads</option>
                  <option value="likes">Likes</option>
                  <option value="trending_score">Trending Score</option>
                </select>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Distribution of {histogramAttribute.replace('_', ' ')} across all models
                </div>
              </label>
            </div>
          )}

          {selectedModels.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.5rem', background: '#e3f2fd', borderRadius: '4px' }}>
              <strong>Selected: {selectedModels.length} models</strong>
              <button
                onClick={() => setSelectedModels([])}
                style={{ marginLeft: '0.5rem', padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
              >
                Clear
              </button>
            </div>
          )}
        </aside>

        <main className="visualization">
          {loading && <div className="loading">Loading models...</div>}
          {error && <div className="error">Error: {error}</div>}
          {!loading && !error && data.length === 0 && (
            <div className="empty">No models match the filters</div>
          )}
          {!loading && !error && data.length > 0 && (
            <>
              {viewMode === '3d' && (
                <div style={{ display: 'flex', gap: '10px', width: '100%', height: '100%' }}>
                  <div 
                    style={{ flex: 1, position: 'relative' }}
                    onMouseMove={(e) => {
                      // Update tooltip position when mouse moves
                      setTooltipPosition({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseLeave={() => {
                      setHoveredModel(null);
                      setTooltipPosition(null);
                    }}
                  >
                    <Suspense fallback={<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#666' }}>Loading 3D visualization...</div>}>
                      <ScatterPlot3D
                        width={width * 0.8}
                        height={height}
                        data={data}
                        familyTree={familyTree.length > 0 ? familyTree : undefined}
                        colorBy={colorBy}
                        sizeBy={sizeBy}
                        colorScheme={colorScheme}
                        showLegend={showLegend}
                        onPointClick={(model) => {
                          setSelectedModel(model);
                          setIsModalOpen(true);
                        }}
                        selectedModelId={familyTreeModelId}
                        onViewChange={setViewCenter}
                        targetViewCenter={viewCenter}
                        onHover={(model, pointer) => {
                          setHoveredModel(model);
                          if (model && pointer) {
                            setTooltipPosition(pointer);
                          } else {
                            setTooltipPosition(null);
                          }
                        }}
                      />
                    </Suspense>
                    <ModelTooltip 
                      model={hoveredModel}
                      position={tooltipPosition}
                      visible={!!hoveredModel && !!tooltipPosition}
                    />
                  </div>
                  <div style={{ width: width * 0.2, height: height, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{
                      width: width * 0.2 - 20,
                      padding: '8px',
                      background: '#f5f5f5',
                      borderRadius: '2px',
                      border: '1px solid #d0d0d0',
                      fontSize: '10px',
                      fontFamily: "'Vend Sans', sans-serif"
                    }}>
                      <h4 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '11px', fontWeight: '600' }}>UV Projection</h4>
                      <p style={{ margin: 0, lineHeight: '1.3', color: '#666', fontSize: '9px' }}>
                        This 2D projection shows the XY plane of the latent space. Click on any point to navigate the 3D view to that region.
                      </p>
                    </div>
                    <UVProjectionSquare
                      width={width * 0.2 - 20}
                      height={height * 0.3}
                      data={data}
                      familyTree={familyTree.length > 0 ? familyTree : undefined}
                      colorBy={colorBy}
                      onRegionSelect={(center) => {
                        setViewCenter(center);
                        // Camera will automatically animate to this position via targetViewCenter prop
                      }}
                      selectedModelId={familyTreeModelId}
                      currentViewCenter={viewCenter}
                    />
                    {viewCenter && (
                      <div style={{
                        width: width * 0.2 - 20,
                        padding: '8px',
                        background: '#f5f5f5',
                        borderRadius: '2px',
                        border: '1px solid #d0d0d0',
                        fontSize: '10px',
                        fontFamily: "'Vend Sans', sans-serif"
                      }}>
                        <strong style={{ fontSize: '10px' }}>View Center:</strong>
                        <div style={{ fontSize: '9px', marginTop: '0.25rem', color: '#666' }}>
                          X: {viewCenter.x.toFixed(3)}<br />
                          Y: {viewCenter.y.toFixed(3)}<br />
                          Z: {viewCenter.z.toFixed(3)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {viewMode === 'scatter' && (
                <EnhancedScatterPlot
                  width={width}
                  height={height}
                  data={data}
                  colorBy={colorBy}
                  sizeBy={sizeBy}
                  onPointClick={(model) => {
                    setSelectedModel(model);
                    setIsModalOpen(true);
                  }}
                  onBrush={(selected) => {
                    setSelectedModels(selected);
                  }}
                />
              )}
              {viewMode === 'network' && (
                <NetworkGraph
                  width={width}
                  height={height}
                  data={data}
                  onNodeClick={(model) => {
                    setSelectedModel(model);
                    setIsModalOpen(true);
                  }}
                />
              )}
              {viewMode === 'histogram' && (
                <Histogram
                  width={width}
                  height={height}
                  data={data}
                  attribute={histogramAttribute}
                />
              )}

              {viewMode === 'paper-plots' && (
                <PaperPlots
                  data={data}
                  width={width}
                  height={height}
                />
              )}
            </>
          )}
        </main>

        <ModelModal
          model={selectedModel}
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onBookmark={selectedModel ? () => toggleBookmark(selectedModel.model_id) : undefined}
          onAddToComparison={selectedModel ? () => addToComparison(selectedModel) : undefined}
          onLoadSimilar={selectedModel ? () => loadSimilarModels(selectedModel.model_id) : undefined}
          isBookmarked={selectedModel ? bookmarkedModels.includes(selectedModel.model_id) : false}
        />
      </div>
    </div>
  );
}

export default App;

