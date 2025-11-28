import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
// Visualizations
import ScatterPlot from './components/visualizations/ScatterPlot';
import ScatterPlot3D from './components/visualizations/ScatterPlot3D';
import NetworkGraph from './components/visualizations/NetworkGraph';
import DistributionView from './components/visualizations/DistributionView';
// Controls
import RandomModelButton from './components/controls/RandomModelButton';
import ZoomSlider from './components/controls/ZoomSlider';
import ThemeToggle from './components/controls/ThemeToggle';
// import RenderingStyleSelector from './components/controls/RenderingStyleSelector';
// import VisualizationModeButtons from './components/controls/VisualizationModeButtons';
// import ClusterFilter from './components/controls/ClusterFilter';
import type { Cluster } from './components/controls/ClusterFilter';
import NodeDensitySlider from './components/controls/NodeDensitySlider';
// Modals
import ModelModal from './components/modals/ModelModal';
// UI Components
import LiveModelCount from './components/ui/LiveModelCount';
// import ModelTooltip from './components/ui/ModelTooltip';
import ErrorBoundary from './components/ui/ErrorBoundary';
import VirtualSearchResults from './components/ui/VirtualSearchResults';
// Types & Utils
import { ModelPoint, Stats, FamilyTree, SearchResult, SimilarModel } from './types';
import cache, { IndexedDBCache } from './utils/data/indexedDB';
import { debounce } from './utils/debounce';
import requestManager from './utils/api/requestManager';
import { fetchWithMsgPack, decodeModelsMsgPack } from './utils/api/msgpackDecoder';
import { useFilterStore, ViewMode, ColorByOption, SizeByOption } from './stores/filterStore';
import { API_BASE } from './config/api';
import './App.css';

const logger = {
  error: (message: string, error?: unknown) => {
    if (process.env.NODE_ENV === 'development') {
      console.error(message, error);
    }
  },
};

function App() {
  // Filter store state
  const {
    viewMode,
    colorBy,
    sizeBy,
    colorScheme,
    showLabels,
    zoomLevel,
    // nodeDensity,
    // renderingStyle,
    theme,
    selectedClusters,
    searchQuery,
    minDownloads,
    minLikes,
    setViewMode,
    setColorBy,
    setSizeBy,
    setColorScheme,
    setShowLabels,
    setZoomLevel,
    // setNodeDensity,
    // setRenderingStyle,
    // setSelectedClusters,
    setSearchQuery,
    setMinDownloads,
    setMinLikes,
    getActiveFilterCount,
    resetFilters: resetFilterStore,
  } = useFilterStore();

  // Initialize theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const [data, setData] = useState<ModelPoint[]>([]);
  const [filteredCount, setFilteredCount] = useState<number | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelPoint | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModels, setSelectedModels] = useState<ModelPoint[]>([]);
  const [baseModelsOnly, setBaseModelsOnly] = useState(false);
  const [semanticSimilarityMode, setSemanticSimilarityMode] = useState(false);
  const [semanticQueryModel, setSemanticQueryModel] = useState<string | null>(null);
  
  const [familyTree, setFamilyTree] = useState<ModelPoint[]>([]);
  const [familyTreeModelId, setFamilyTreeModelId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchInput, setSearchInput] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);
  // const [viewCenter, setViewCenter] = useState<{ x: number; y: number; z: number } | null>(null);
  const [projectionMethod, setProjectionMethod] = useState<'umap' | 'tsne'>('umap');
  const [bookmarkedModels, setBookmarkedModels] = useState<string[]>([]);
  const [comparisonModels, setComparisonModels] = useState<ModelPoint[]>([]);
  const [similarModels, setSimilarModels] = useState<SimilarModel[]>([]);
  const [showSimilar, setShowSimilar] = useState(false);
  const [showLegend, setShowLegend] = useState(true);
  const [hoveredModel, setHoveredModel] = useState<ModelPoint | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  
  // Structural visualization options
  const [showNetworkEdges, setShowNetworkEdges] = useState(false);
  const [showStructuralGroups, setShowStructuralGroups] = useState(false);
  const [overviewMode, setOverviewMode] = useState(false);
  const [networkEdgeType, setNetworkEdgeType] = useState<'library' | 'pipeline' | 'combined'>('combined');
  const [maxHierarchyDepth, setMaxHierarchyDepth] = useState<number | null>(null);
  const [showDistanceHeatmap, setShowDistanceHeatmap] = useState(false);
  const [highlightedPath, setHighlightedPath] = useState<string[]>([]);
  const [useGraphEmbeddings, setUseGraphEmbeddings] = useState(false);
  const [embeddingType, setEmbeddingType] = useState<string>('text-only');
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [clustersLoading, setClustersLoading] = useState(false);
  
  const activeFilterCount = getActiveFilterCount();
  
  const resetFilters = useCallback(() => {
    resetFilterStore();
    setMinDownloads(0);
    setMinLikes(0);
    setSearchQuery('');
  }, [resetFilterStore, setMinDownloads, setMinLikes, setSearchQuery]);

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
      if (cachedModels && cachedModels.length > 0) {
        setData(cachedModels);
        setFilteredCount(cachedModels.length);
        setLoading(false);
        // Fetch in background to update cache if stale
        setTimeout(() => {
          fetchData();
        }, 100);
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
        // Try MessagePack first, fallback to JSON
        try {
          const result = await fetchWithMsgPack<{ models: ModelPoint[] }>(url);
          models = result.models || [];
          count = models.length;
        } catch (msgpackError) {
          // Fallback to JSON
          const response = await requestManager.fetch(url, {}, cacheKey);
          if (!response.ok) throw new Error('Failed to fetch similar models');
          const result = await response.json();
          models = result.models || [];
          count = models.length;
        }
      } else {
        const params = new URLSearchParams({
          min_downloads: minDownloads.toString(),
          min_likes: minLikes.toString(),
          color_by: colorBy,
          size_by: sizeBy,
          projection_method: projectionMethod,
          base_models_only: baseModelsOnly.toString(),
          use_graph_embeddings: useGraphEmbeddings.toString(),
        });
        if (searchQuery) {
          params.append('search_query', searchQuery);
        }
        
        params.append('max_points', viewMode === '3d' ? '50000' : viewMode === 'scatter' ? '10000' : viewMode === 'network' ? '500' : '5000');
        // Add format parameter for MessagePack support
        params.append('format', 'msgpack');

        const url = `${API_BASE}/api/models?${params}`;
        // Try MessagePack first for better performance, fallback to JSON
        try {
          const response = await requestManager.fetch(url, {
            headers: {
              'Accept': 'application/msgpack',
            },
          }, cacheKey);
          
          if (!response.ok) throw new Error('Failed to fetch models');
          
          const contentType = response.headers.get('content-type');
          if (contentType?.includes('application/msgpack')) {
            // Decode MessagePack binary response (backend returns compact format array)
            const buffer = await response.arrayBuffer();
            models = decodeModelsMsgPack(new Uint8Array(buffer));
            count = models.length;
            setEmbeddingType('text-only'); // MessagePack response doesn't include metadata
          } else {
            // Response was JSON (backend may not support msgpack or returned JSON)
            const result = await response.json();
            if (Array.isArray(result)) {
              models = result;
              count = models.length;
              setEmbeddingType('text-only');
            } else {
              models = result.models || [];
              count = result.filtered_count ?? models.length;
              setEmbeddingType(result.embedding_type || 'text-only');
            }
          }
        } catch (error) {
          // Fallback to JSON if MessagePack fails
          const jsonUrl = url.replace('format=msgpack', 'format=json');
          const response = await requestManager.fetch(jsonUrl, {}, cacheKey);
          if (!response.ok) throw new Error('Failed to fetch models');
          const result = await response.json();
          
          if (Array.isArray(result)) {
            models = result;
            count = models.length;
            setEmbeddingType('text-only');
          } else {
            models = result.models || [];
            count = result.filtered_count ?? models.length;
            setEmbeddingType(result.embedding_type || 'text-only');
          }
        }
      }
      
      await cache.cacheModels(cacheKey, models);
      
      setData(models);
      setFilteredCount(count);
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        // Check if it's a connection error (backend not ready)
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
          setError('Backend is starting up. Please wait... The first load may take a few minutes.');
        } else {
          setError(errorMessage);
        }
      }
    } finally {
      setLoading(false);
      fetchDataAbortRef.current = null;
    }
  }, [minDownloads, minLikes, searchQuery, colorBy, sizeBy, projectionMethod, baseModelsOnly, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, selectedClusters, viewMode]);

  const debouncedFetchData = useMemo(
    () => debounce(fetchData, 300),
    [fetchData]
  );

  // Consolidated effect to handle both search and filter changes
  useEffect(() => {
    // For search queries, use debounced version
    if (searchQuery) {
      debouncedFetchData();
      return () => {
        debouncedFetchData.cancel();
      };
    } else {
      // For filter changes without search, also use debounced version
      debouncedFetchData();
      return () => {
        debouncedFetchData.cancel();
      };
    }
  }, [searchQuery, minDownloads, minLikes, colorBy, sizeBy, baseModelsOnly, projectionMethod, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, selectedClusters, viewMode, debouncedFetchData]);

  // Function to clear cache and refresh stats
  const clearCacheAndRefresh = useCallback(async () => {
    try {
      // Clear all caches
      await cache.clear('stats');
      await cache.clear('models');
      console.log('Cache cleared successfully');
      
      // Immediately fetch fresh stats
      const response = await fetch(`${API_BASE}/api/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const statsData = await response.json();
      await cache.cacheStats('stats', statsData);
      setStats(statsData);
      
      // Refresh model data
      fetchData();
    } catch (err) {
      if (err instanceof Error) {
        logger.error('Error clearing cache:', err);
      }
    }
  }, [fetchData]);

  useEffect(() => {
    const fetchStats = async () => {
      const cacheKey = 'stats';
      const cachedStats = await cache.getCachedStats(cacheKey);
      
      // Always fetch fresh stats on initial load, ignore stale cache
      // This fixes the issue of showing old data (1000 models) when backend has 5000
      if (cachedStats) {
        // Show cached data temporarily
        setStats(cachedStats);
      }
      
      // Always fetch fresh stats to update
      try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        const statsData = await response.json();
        await cache.cacheStats(cacheKey, statsData);
        setStats(statsData);
      } catch (err) {
        if (err instanceof Error) {
          logger.error('Error fetching stats:', err);
        }
      }
    };

    fetchStats();
  }, []);

  // Fetch clusters
  useEffect(() => {
    const fetchClusters = async () => {
      setClustersLoading(true);
      try {
        const response = await fetch(`${API_BASE}/api/clusters`);
        if (!response.ok) throw new Error('Failed to fetch clusters');
        const data = await response.json();
        setClusters(data.clusters || []);
      } catch (err) {
        logger.error('Error fetching clusters:', err);
        setClusters([]);
      } finally {
        setClustersLoading(false);
      }
    };

    fetchClusters();
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
      logger.error('Search error:', err);
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
      logger.error('Family tree error:', err);
      setFamilyTree([]);
      setFamilyTreeModelId(null);
    }
  }, []);

  const clearFamilyTree = useCallback(() => {
    setFamilyTree([]);
    setFamilyTreeModelId(null);
  }, []);

  const loadFamilyPath = useCallback(async (modelId: string, targetId?: string) => {
    try {
      const url = targetId
        ? `${API_BASE}/api/family/path/${encodeURIComponent(modelId)}?target_id=${encodeURIComponent(targetId)}`
        : `${API_BASE}/api/family/path/${encodeURIComponent(modelId)}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to load path');
      const data = await response.json();
      setHighlightedPath(data.path || []);
    } catch (err) {
      logger.error('Path loading error:', err);
      setHighlightedPath([]);
    }
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
      logger.error('Similar models error:', err);
      if (errorMessage !== 'Failed to load similar models' || !(err instanceof TypeError && err.message.includes('fetch'))) {
        setError(`Similar models: ${errorMessage}`);
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
      logger.error('Export error:', err);
      alert('Failed to export models');
    }
  }, []);

  return (
    <ErrorBoundary>
      <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', width: '100%' }}>
          <div style={{ flex: '1 1 auto', minWidth: '250px' }}>
            <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '600', lineHeight: '1.2' }}>ML Ecosystem: 2M Models on Hugging Face</h1>
            <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', opacity: 0.9, display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <a href="https://arxiv.org/abs/2508.06811" target="_blank" rel="noopener noreferrer" style={{ color: '#64b5f6', textDecoration: 'none', whiteSpace: 'nowrap' }}>Paper</a>
              <a href="https://github.com/bendlaufer/ai-ecosystem" target="_blank" rel="noopener noreferrer" style={{ color: '#64b5f6', textDecoration: 'none', whiteSpace: 'nowrap' }}>GitHub</a>
              <a href="https://huggingface.co/modelbiome" target="_blank" rel="noopener noreferrer" style={{ color: '#64b5f6', textDecoration: 'none', whiteSpace: 'nowrap' }}>Dataset</a>
              <span style={{ opacity: 0.7, whiteSpace: 'nowrap' }}>Laufer, Oderinwale, Kleinberg</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', flexShrink: 0 }}>
            <LiveModelCount compact={true} />
            {stats && (
              <>
                <div className="stats" style={{ display: 'flex', gap: '0.5rem', fontSize: '0.8rem', flexWrap: 'wrap' }}>
                  <span>{stats.total_models.toLocaleString()} models</span>
                  <span>{stats.unique_libraries} libraries</span>
                </div>
                <button
                  onClick={clearCacheAndRefresh}
                  style={{
                    background: 'rgba(255, 255, 255, 0.15)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    borderRadius: '0',
                    width: '32px',
                    height: '32px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '16px',
                    transition: 'all 0.2s ease',
                    flexShrink: 0,
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.25)';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                  }}
                  title="Refresh data and clear cache"
                  aria-label="Refresh data"
                >
                  ⟳
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            marginBottom: '1.5rem',
            paddingBottom: '1rem',
            borderBottom: '1px solid #e0e0e0'
          }}>
            <h2 style={{ 
              margin: 0,
              fontSize: '1.5rem',
              fontWeight: '600',
              color: '#2d2d2d'
            }}>
              Filters & Controls
            </h2>
            {activeFilterCount > 0 && (
              <div style={{ 
                fontSize: '0.75rem', 
                background: '#4a4a4a',
                color: 'white', 
                padding: '0.35rem 0.7rem', 
                borderRadius: '0',
                fontWeight: '600'
              }}>
                {activeFilterCount} active
              </div>
            )}
          </div>
          
          {/* Filter Results Count */}
          {!loading && data.length > 0 && (
            <div className="sidebar-section" style={{ 
              background: '#f5f5f5',
              border: '1px solid #d0d0d0',
              fontSize: '0.9rem',
              marginBottom: '1.5rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <div>
                  <strong style={{ fontSize: '1.1rem', color: '#2d2d2d' }}>
                    {data.length.toLocaleString()}
                  </strong>
                  <span style={{ marginLeft: '0.4rem', color: '#4a4a4a' }}>
                    {data.length === 1 ? 'model' : 'models'}
                  </span>
                </div>
                {embeddingType === 'graph-aware' && (
                  <span style={{ 
                    fontSize: '0.7rem', 
                    background: '#4a4a4a',
                    color: 'white', 
                    padding: '0.3rem 0.6rem', 
                    borderRadius: '0',
                    fontWeight: '600'
                  }}>
                    Graph
                  </span>
                )}
              </div>
              {filteredCount !== null && filteredCount !== data.length && (
                <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.25rem' }}>
                  of {filteredCount.toLocaleString()} matching
                </div>
              )}
              {stats && filteredCount !== null && filteredCount < stats.total_models && (
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  from {stats.total_models.toLocaleString()} total
                </div>
              )}
            </div>
          )}

          {/* Search Section */}
          <div className="sidebar-section">
            <h3>Search</h3>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models, tags, libraries..."
              style={{ width: '100%' }}
              title="Search by model name, tags, library, or metadata"
            />
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

          {/* License Filter - Collapsed */}
          {stats && stats.licenses && typeof stats.licenses === 'object' && Object.keys(stats.licenses).length > 0 && (
            <details className="sidebar-section" style={{ border: '1px solid #e0e0e0', borderRadius: '0', padding: '0.75rem' }}>
              <summary style={{ cursor: 'pointer', fontWeight: '600', fontSize: '0.95rem', listStyle: 'none', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span>Licenses ({Object.keys(stats.licenses).length})</span>
              </summary>
              <div style={{ maxHeight: '200px', overflowY: 'auto', marginTop: '1rem' }}>
                {Object.entries(stats.licenses as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 20)
                  .map(([license, count]) => (
                    <label 
                      key={license}
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.4rem', 
                        marginBottom: '0.4rem',
                        cursor: 'pointer',
                        fontSize: '0.85rem'
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={searchQuery.toLowerCase().includes(license.toLowerCase())}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSearchQuery(searchQuery ? `${searchQuery} ${license}` : license);
                          } else {
                            setSearchQuery(searchQuery.replace(license, '').trim() || '');
                          }
                        }}
                      />
                      <span style={{ flex: 1 }}>{license || 'Unknown'}</span>
                      <span style={{ fontSize: '0.7rem', color: '#999' }}>({Number(count).toLocaleString()})</span>
                    </label>
                  ))}
              </div>
            </details>
          )}

          {/* Quick Actions - Consolidated */}
          <div className="sidebar-section">
            <h3>Quick Actions</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <RandomModelButton
                data={data}
                onSelect={(model: ModelPoint) => {
                  setSelectedModel(model);
                  setIsModalOpen(true);
                }}
                disabled={loading || data.length === 0}
              />
              <button
                onClick={() => {
                  const avgDownloads = data.reduce((sum, m) => sum + (m.downloads || 0), 0) / data.length;
                  setMinDownloads(Math.floor(avgDownloads));
                }}
                disabled={loading || data.length === 0}
                style={{
                  padding: '0.75rem',
                  background: '#4a90e2',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0',
                  cursor: loading || data.length === 0 ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  fontSize: '0.9rem',
                  opacity: loading || data.length === 0 ? 0.5 : 1
                }}
                title="Filter to models with above average downloads"
              >
                Popular Models
              </button>
              <button
                onClick={resetFilters}
                style={{
                  padding: '0.75rem',
                  background: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '0.9rem'
                }}
                title="Clear all filters and reset to defaults"
              >
                Reset All
              </button>
            </div>
          </div>

          {/* Visualization */}
          <div className="sidebar-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Visualization</h3>
              <ThemeToggle />
            </div>
            
            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>View Mode</span>
              <select 
                value={viewMode} 
                onChange={(e) => setViewMode(e.target.value as ViewMode)}
                style={{ width: '100%', padding: '0.6rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.9rem' }}
                title="Choose how to visualize the models"
              >
                <option value="3d">3D Scatter</option>
                <option value="scatter">2D Scatter</option>
                <option value="network">Network</option>
                <option value="distribution">Distribution</option>
              </select>
            </label>

            {/* Zoom and Label Controls for Scatter View */}
            {viewMode === 'scatter' && (
              <>
                <ZoomSlider
                  value={zoomLevel}
                  onChange={setZoomLevel}
                  min={0.1}
                  max={5}
                  step={0.1}
                  disabled={loading}
                />
                <NodeDensitySlider disabled={loading} />
                <div className="label-toggle">
                  <span className="label-toggle-label">Show Labels</span>
                  <label className="label-toggle-switch">
                    <input
                      type="checkbox"
                      checked={showLabels}
                      onChange={(e) => setShowLabels(e.target.checked)}
                    />
                    <span className="label-toggle-slider"></span>
                  </label>
                </div>
              </>
            )}

            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Color By</span>
              <select 
                value={colorBy} 
                onChange={(e) => setColorBy(e.target.value as ColorByOption)}
                style={{ width: '100%', padding: '0.6rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.9rem' }}
                title="Choose what attribute to color models by"
              >
                <option value="library_name">Library</option>
                <option value="pipeline_tag">Pipeline/Task</option>
                <option value="cluster_id">Cluster</option>
                <option value="family_depth">Family Depth</option>
                <option value="downloads">Downloads</option>
                <option value="likes">Likes</option>
                <option value="trending_score">Trending</option>
                <option value="licenses">License</option>
              </select>
            </label>

            {/* Color Scheme */}
            {(colorBy === 'downloads' || colorBy === 'likes' || colorBy === 'family_depth' || colorBy === 'trending_score') && (
              <label style={{ marginBottom: '1rem', display: 'block' }}>
                <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Color Scheme</span>
                <select 
                  value={colorScheme} 
                  onChange={(e) => setColorScheme(e.target.value as any)}
                  style={{ width: '100%', padding: '0.6rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.9rem' }}
                >
                  <option value="viridis">Viridis</option>
                  <option value="plasma">Plasma</option>
                  <option value="inferno">Inferno</option>
                  <option value="magma">Magma</option>
                  <option value="coolwarm">Cool-Warm</option>
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
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Size By</span>
              <select 
                value={sizeBy} 
                onChange={(e) => setSizeBy(e.target.value as SizeByOption)}
                style={{ width: '100%', padding: '0.6rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.9rem' }}
                title="Choose what determines point size"
              >
                <option value="downloads">Downloads</option>
                <option value="likes">Likes</option>
                <option value="none">Uniform</option>
              </select>
            </label>

            <details style={{ marginBottom: '1rem', marginTop: '1rem' }}>
              <summary style={{ cursor: 'pointer', fontWeight: '500', fontSize: '0.9rem', marginBottom: '0rem' }}>
                Advanced Settings
              </summary>
              <div style={{ marginTop: '0.75rem' }}>
                <select 
                  value={projectionMethod} 
                  onChange={(e) => setProjectionMethod(e.target.value as 'umap' | 'tsne')}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.85rem' }}
                  title="UMAP preserves global structure, t-SNE emphasizes local clusters"
                >
                  <option value="umap">UMAP</option>
                  <option value="tsne">t-SNE</option>
                </select>
              </div>
            </details>
          </div>

          {/* Display Options - Simplified */}
          <details className="sidebar-section" open>
            <summary style={{ cursor: 'pointer', fontWeight: '600', fontSize: '1rem', marginBottom: '1rem', listStyle: 'none' }}>
              <h3 style={{ display: 'inline', margin: 0 }}>Display Options</h3>
            </summary>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} title="Show only root models without parents">
                <input
                  type="checkbox"
                  checked={baseModelsOnly}
                  onChange={(e) => setBaseModelsOnly(e.target.checked)}
                  style={{ marginRight: '0.5rem', cursor: 'pointer' }}
                />
                <span style={{ fontWeight: '500', fontSize: '0.9rem' }}>Base Models Only</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} title="Use family tree structure in embeddings">
                <input
                  type="checkbox"
                  checked={useGraphEmbeddings}
                  onChange={(e) => setUseGraphEmbeddings(e.target.checked)}
                  style={{ marginRight: '0.5rem', cursor: 'pointer' }}
                />
                <span style={{ fontWeight: '500', fontSize: '0.9rem' }}>Graph-Aware Layout</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} title="Sort by similarity to a specific model">
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
                <span style={{ fontWeight: '500', fontSize: '0.9rem' }}>Similarity View</span>
              </label>
            </div>

            {semanticSimilarityMode && (
              <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f9f9f9', borderRadius: '0', border: '1px solid #e0e0e0' }}>
                <input
                  type="text"
                  value={semanticQueryModel || ''}
                  onChange={(e) => setSemanticQueryModel(e.target.value || null)}
                  placeholder="Enter model ID..."
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.85rem' }}
                  title="Enter a model ID to compare against"
                />
                {selectedModel && (
                  <button
                    onClick={() => setSemanticQueryModel(selectedModel.model_id)}
                    style={{
                      marginTop: '0.5rem',
                      padding: '0.4rem 0.7rem',
                      background: '#4a90e2',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      width: '100%'
                    }}
                    title="Use the currently selected model"
                  >
                    Use Selected Model
                  </button>
                )}
              </div>
            )}
          </details>

          {/* Structural Visualization Options */}
          {viewMode === 'network' && (
            <div className="sidebar-section">
              <h3>Network Structure</h3>
              <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '1rem', lineHeight: '1.4' }}>
                Explore relationships and structure in the model ecosystem
              </div>
              
              <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={overviewMode}
                  onChange={(e) => setOverviewMode(e.target.checked)}
                  style={{ marginRight: '0.5rem', cursor: 'pointer' }}
                />
              <div>
                <span style={{ fontWeight: '500' }}>Overview Mode</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Zoom out to see full ecosystem structure with all relationships visible. Camera will automatically adjust.
                </div>
              </div>
              </label>

              <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showNetworkEdges}
                  onChange={(e) => setShowNetworkEdges(e.target.checked)}
                  style={{ marginRight: '0.5rem', cursor: 'pointer' }}
                />
              <div>
                <span style={{ fontWeight: '500' }}>Network Relationships</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Show connections between related models (same library, pipeline, or tags). Blue = library, Pink = pipeline.
                </div>
              </div>
              </label>

              {showNetworkEdges && (
                <div style={{ marginLeft: '1.5rem', marginBottom: '1rem', padding: '0.75rem', background: 'white', borderRadius: '0', border: '1px solid #d0d0d0' }}>
                  <label style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                    Connection Type
                  </label>
                  <select 
                    value={networkEdgeType} 
                    onChange={(e) => setNetworkEdgeType(e.target.value as 'library' | 'pipeline' | 'combined')}
                    style={{ width: '100%', padding: '0.5rem', borderRadius: '0', border: '1px solid #d0d0d0', fontSize: '0.85rem' }}
                  >
                    <option value="combined">Combined (library + pipeline + tags)</option>
                    <option value="library">Library Only</option>
                    <option value="pipeline">Pipeline Only</option>
                  </select>
                </div>
              )}

              <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showStructuralGroups}
                  onChange={(e) => setShowStructuralGroups(e.target.checked)}
                  style={{ marginRight: '0.5rem', cursor: 'pointer' }}
                />
              <div>
                <span style={{ fontWeight: '500' }}>Structural Groupings</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Highlight clusters and groups with wireframe boundaries. Shows top library and pipeline clusters.
                </div>
              </div>
              </label>
            </div>
          )}

          {/* Advanced Hierarchy Controls */}
          <details className="sidebar-section" style={{ border: '1px solid #e0e0e0', borderRadius: '0', padding: '0.75rem' }}>
            <summary style={{ cursor: 'pointer', fontWeight: '600', fontSize: '0.95rem', listStyle: 'none' }}>
              Hierarchy & Structure
            </summary>
            <div style={{ marginTop: '1rem' }}>
              <label style={{ marginBottom: '1rem', display: 'block' }}>
                <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                  Max Hierarchy Depth
                </span>
                <input
                  type="range"
                  min="0"
                  max="10"
                  value={maxHierarchyDepth ?? 10}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setMaxHierarchyDepth(val === 10 ? null : val);
                  }}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span>All levels</span>
                  <span>{maxHierarchyDepth !== null ? `Depth ≤ ${maxHierarchyDepth}` : 'No limit'}</span>
                </div>
              </label>
              <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={showDistanceHeatmap}
                  onChange={(e) => setShowDistanceHeatmap(e.target.checked)}
                />
                <span style={{ fontSize: '0.85rem' }}>Distance Heatmap</span>
              </label>
              {selectedModel && (
                <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: '#f5f5f5', borderRadius: '0', fontSize: '0.85rem' }}>
                  <div style={{ fontWeight: '500', marginBottom: '0.25rem' }}>Selected Model:</div>
                  <div style={{ color: '#666', marginBottom: '0.5rem', wordBreak: 'break-word' }}>{selectedModel.model_id}</div>
                  {selectedModel.family_depth !== null && (
                    <div style={{ color: '#666', marginBottom: '0.5rem' }}>
                      Hierarchy Depth: {selectedModel.family_depth}
                    </div>
                  )}
                  <button
                    onClick={() => {
                      if (selectedModel.parent_model) {
                        loadFamilyPath(selectedModel.model_id, selectedModel.parent_model);
                      } else {
                        loadFamilyPath(selectedModel.model_id);
                      }
                    }}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.8rem',
                      background: '#4a90e2',
                      color: 'white',
                      border: 'none',
                      borderRadius: '0',
                      cursor: 'pointer',
                      marginRight: '0.5rem',
                      marginBottom: '0.5rem'
                    }}
                  >
                    Show Path to Root
                  </button>
                  <button
                    onClick={() => setHighlightedPath([])}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.8rem',
                      background: '#6a6a6a',
                      color: 'white',
                      border: 'none',
                      borderRadius: '0',
                      cursor: 'pointer',
                      marginBottom: '0.5rem'
                    }}
                  >
                    Clear Path
                  </button>
                </div>
              )}
            </div>
          </details>

          <div className="sidebar-section">
            <h3>Family Tree Explorer</h3>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onFocus={() => searchInput.length > 0 && setShowSearchResults(true)}
                placeholder="Type model name..."
                style={{ width: '100%', padding: '0.5rem', borderRadius: '0', border: '1px solid #d0d0d0' }}
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
                  maxHeight: '400px',
                  zIndex: 1000,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  <VirtualSearchResults
                    results={searchResults}
                    onSelect={(result) => {
                      loadFamilyTree(result.model_id);
                    }}
                  />
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
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '0', border: '1px solid #d0d0d0' }}>
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
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '0', border: '1px solid #d0d0d0' }}>
              <h3 style={{ marginTop: 0, fontSize: '0.9rem', fontWeight: '600' }}>Comparison ({comparisonModels.length}/3)</h3>
              {comparisonModels.map(model => (
                <div key={model.model_id} style={{ marginBottom: '0.5rem', padding: '0.5rem', background: 'white', borderRadius: '0', fontSize: '0.85rem' }}>
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
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '0', border: '1px solid #d0d0d0' }}>
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
                  <div key={idx} style={{ marginBottom: '0.5rem', padding: '0.5rem', background: 'white', borderRadius: '0' }}>
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


          {selectedModels.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.5rem', background: '#e3f2fd', borderRadius: '0' }}>
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
              {viewMode === 'scatter' && (
                <ScatterPlot
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
              {viewMode === '3d' && (
                <ScatterPlot3D
                  data={data}
                  colorBy={colorBy}
                  sizeBy={sizeBy}
                  hoveredModel={hoveredModel}
                  onPointClick={(model) => {
                    setSelectedModel(model);
                    setIsModalOpen(true);
                  }}
                  onHover={(model, position) => {
                    setHoveredModel(model);
                    if (model && position) {
                      setTooltipPosition(position);
                    } else {
                      setTooltipPosition(null);
                    }
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
              {viewMode === 'distribution' && (
                <DistributionView data={data} width={width} height={height} />
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
    </ErrorBoundary>
  );
}

export default App;

