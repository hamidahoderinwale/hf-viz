import React, { useState, useEffect, useCallback, useRef, useMemo, lazy, Suspense } from 'react';
// Visualizations
import EnhancedScatterPlot from './components/visualizations/EnhancedScatterPlot';
import NetworkGraph from './components/visualizations/NetworkGraph';
import UVProjectionSquare from './components/visualizations/UVProjectionSquare';
import DistributionView from './components/visualizations/DistributionView';
import StackedView from './components/visualizations/StackedView';
import HeatmapView from './components/visualizations/HeatmapView';
// Controls
import RandomModelButton from './components/controls/RandomModelButton';
import ZoomSlider from './components/controls/ZoomSlider';
import ThemeToggle from './components/controls/ThemeToggle';
import RenderingStyleSelector from './components/controls/RenderingStyleSelector';
import VisualizationModeButtons from './components/controls/VisualizationModeButtons';
import ClusterFilter, { Cluster } from './components/controls/ClusterFilter';
import NodeDensitySlider from './components/controls/NodeDensitySlider';
// Modals
import ModelModal from './components/modals/ModelModal';
// UI Components
import LiveModelCount from './components/ui/LiveModelCount';
import ModelTooltip from './components/ui/ModelTooltip';
import ErrorBoundary from './components/ui/ErrorBoundary';
// Types & Utils
import { ModelPoint, Stats, FamilyTree, SearchResult, SimilarModel } from './types';
import cache, { IndexedDBCache } from './utils/data/indexedDB';
import { debounce } from './utils/debounce';
import requestManager from './utils/api/requestManager';
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

const ScatterPlot3D = lazy(() => import('./components/visualizations/ScatterPlot3D'));

function App() {
  // Filter store state
  const {
    viewMode,
    colorBy,
    sizeBy,
    colorScheme,
    showLabels,
    zoomLevel,
    nodeDensity,
    renderingStyle,
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
    setNodeDensity,
    setRenderingStyle,
    setSelectedClusters,
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
  const [viewCenter, setViewCenter] = useState<{ x: number; y: number; z: number } | null>(null);
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
        const response = await requestManager.fetch(url, {}, cacheKey);
        if (!response.ok) throw new Error('Failed to fetch similar models');
        const result = await response.json();
        models = result.models || [];
        count = models.length;
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
        
        params.append('max_points', '500000');

        const url = `${API_BASE}/api/models?${params}`;
        const response = await requestManager.fetch(url, {}, cacheKey);
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
  }, [minDownloads, minLikes, searchQuery, colorBy, sizeBy, projectionMethod, baseModelsOnly, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, selectedClusters]);

  const debouncedFetchData = useMemo(
    () => debounce(fetchData, 300),
    [fetchData]
  );

  useEffect(() => {
    if (searchQuery) {
      debouncedFetchData();
    } else {
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
  }, [minDownloads, minLikes, colorBy, sizeBy, baseModelsOnly, projectionMethod, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, debouncedFetchData]);

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
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            marginBottom: '1.5rem',
            paddingBottom: '1rem',
            borderBottom: '2px solid #e8e8e8'
          }}>
            <h2 style={{ 
              margin: 0,
              fontSize: '1.5rem',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #5e35b1 0%, #7b1fa2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}>
              Filters & Controls
            </h2>
            {activeFilterCount > 0 && (
              <div style={{ 
                fontSize: '0.75rem', 
                background: 'linear-gradient(135deg, #5e35b1 0%, #7b1fa2 100%)',
                color: 'white', 
                padding: '0.4rem 0.75rem', 
                borderRadius: '16px',
                fontWeight: '600',
                boxShadow: '0 2px 6px rgba(94, 53, 177, 0.3)'
              }}>
                {activeFilterCount} active
              </div>
            )}
          </div>
          
          {/* Filter Results Count */}
          {!loading && data.length > 0 && (
            <div className="sidebar-section" style={{ 
              background: 'linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)',
              border: '2px solid #ce93d8',
              fontSize: '0.9rem',
              marginBottom: '1.5rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <div>
                  <strong style={{ fontSize: '1.1rem', color: '#6a1b9a' }}>
                    {data.length.toLocaleString()}
                  </strong>
                  <span style={{ marginLeft: '0.4rem', color: '#4a148c' }}>
                    {data.length === 1 ? 'model' : 'models'}
                  </span>
                </div>
                {embeddingType === 'graph-aware' && (
                  <span style={{ 
                    fontSize: '0.7rem', 
                    background: '#7b1fa2',
                    color: 'white', 
                    padding: '0.3rem 0.6rem', 
                    borderRadius: '12px',
                    fontWeight: '600'
                  }}>
                    🌐 Graph
                  </span>
                )}
              </div>
              {filteredCount !== null && filteredCount !== data.length && (
                <div style={{ fontSize: '0.8rem', color: '#6a1b9a', marginTop: '0.25rem' }}>
                  of {filteredCount.toLocaleString()} matching
                </div>
              )}
              {stats && filteredCount !== null && filteredCount < stats.total_models && (
                <div style={{ fontSize: '0.75rem', color: '#8e24aa', marginTop: '0.25rem' }}>
                  from {stats.total_models.toLocaleString()} total
                </div>
              )}
            </div>
          )}

          {/* Search Section */}
          <div className="sidebar-section">
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1',
              marginBottom: '0.75rem'
            }}>
              🔍 Search Models
            </h3>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by model ID, tags, or keywords..."
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem', lineHeight: '1.4' }}>
              Search by model name, tags, library, or metadata
            </div>
          </div>

          {/* Popularity Filters */}
          <div className="sidebar-section">
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              📊 Popularity Filters
            </h3>
            
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

          {/* License Filter */}
          {stats && stats.licenses && typeof stats.licenses === 'object' && Object.keys(stats.licenses).length > 0 && (
            <div className="sidebar-section">
              <h3>License Filter</h3>
              <div style={{ maxHeight: '200px', overflowY: 'auto', marginTop: '0.5rem' }}>
                {Object.entries(stats.licenses as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])  // Sort by count descending
                  .slice(0, 20)  // Show top 20 licenses
                  .map(([license, count]) => (
                    <label 
                      key={license}
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.5rem', 
                        marginBottom: '0.5rem',
                        cursor: 'pointer',
                        fontSize: '0.9rem'
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={searchQuery.toLowerCase().includes(license.toLowerCase())}
                        onChange={(e) => {
                          if (e.target.checked) {
                            // Add license to search (simple implementation)
                            setSearchQuery(searchQuery ? `${searchQuery} ${license}` : license);
                          } else {
                            // Remove license from search
                            setSearchQuery(searchQuery.replace(license, '').trim() || '');
                          }
                        }}
                      />
                      <span style={{ flex: 1 }}>{license || 'Unknown'}</span>
                      <span style={{ fontSize: '0.75rem', color: '#666' }}>({Number(count).toLocaleString()})</span>
                    </label>
                  ))}
              </div>
              {Object.keys(stats.licenses).length > 20 && (
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem' }}>
                  Showing top 20 licenses
                </div>
              )}
            </div>
          )}

          {/* Discovery */}
          <div className="sidebar-section">
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              🎲 Discovery
            </h3>
            <RandomModelButton
              data={data}
              onSelect={(model: ModelPoint) => {
                setSelectedModel(model);
                setIsModalOpen(true);
              }}
              disabled={loading || data.length === 0}
            />
          </div>

          {/* Visualization Options */}
          <div className="sidebar-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ 
                margin: 0,
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem',
                color: '#5e35b1'
              }}>
                🎨 Visualization
              </h3>
              <ThemeToggle />
            </div>
            
            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>View Mode</span>
              <select 
                value={viewMode} 
                onChange={(e) => setViewMode(e.target.value as ViewMode)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0' }}
              >
                <option value="3d">3D Latent Space</option>
                <option value="scatter">2D Latent Space</option>
                <option value="network">Network Graph</option>
                <option value="distribution">Distribution</option>
                <option value="stacked">Stacked</option>
                <option value="heatmap">Heatmap</option>
              </select>
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                {viewMode === '3d' && 'Interactive 3D exploration of model relationships'}
                {viewMode === 'scatter' && '2D projection showing model similarity'}
                {viewMode === 'network' && 'Network graph of model connections'}
                {viewMode === 'distribution' && 'Statistical distributions of model properties'}
                {viewMode === 'stacked' && 'Hierarchical view of model families'}
                {viewMode === 'heatmap' && 'Density heatmap in latent space'}
              </div>
            </label>

            {/* Rendering Style Selector for 3D View */}
            {viewMode === '3d' && (
              <div style={{ marginBottom: '1rem' }}>
                <RenderingStyleSelector />
              </div>
            )}

            {/* Zoom and Label Controls for 3D View */}
            {viewMode === '3d' && (
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
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>Color Encoding</span>
              <select 
                value={colorBy} 
                onChange={(e) => setColorBy(e.target.value as ColorByOption)}
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
                onChange={(e) => setSizeBy(e.target.value as SizeByOption)}
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
          <div className="sidebar-section" style={{ background: 'linear-gradient(135deg, #f3e5f5 0%, #fce4ec 100%)', border: '2px solid #f48fb1' }}>
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              ⚡ View Modes
            </h3>
            
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

            <label style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={useGraphEmbeddings}
                onChange={(e) => setUseGraphEmbeddings(e.target.checked)}
                style={{ marginRight: '0.5rem', cursor: 'pointer' }}
              />
              <div>
                <span style={{ fontWeight: '500' }}>🌐 Graph-Aware Embeddings</span>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Use embeddings that respect family tree structure. Models in the same family will be closer together.
                </div>
              </div>
            </label>
            
            {embeddingType && (
              <div style={{ 
                marginTop: '0.5rem', 
                padding: '0.75rem', 
                background: embeddingType === 'graph-aware' ? '#e8f5e9' : '#f5f5f5',
                border: `1px solid ${embeddingType === 'graph-aware' ? '#4caf50' : '#d0d0d0'}`,
                borderRadius: '4px',
                fontSize: '0.75rem',
                color: '#666'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <strong style={{ color: embeddingType === 'graph-aware' ? '#2e7d32' : '#666' }}>
                    {embeddingType === 'graph-aware' ? '🌐 Graph-Aware' : '📝 Text-Only'} Embeddings
                  </strong>
                </div>
                <div style={{ fontSize: '0.7rem', color: '#888', lineHeight: '1.4' }}>
                  {embeddingType === 'graph-aware' 
                    ? 'Models in the same family tree are positioned closer together, revealing hierarchical relationships.'
                    : 'Standard text-based embeddings showing semantic similarity from model descriptions and tags.'}
                </div>
              </div>
            )}

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

          {/* Structural Visualization Options */}
          {viewMode === '3d' && (
            <div className="sidebar-section" style={{ background: 'linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%)', border: '2px solid #aed581' }}>
              <h3 style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem',
                color: '#5e35b1'
              }}>
                🔗 Network Structure
              </h3>
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
                  <span style={{ fontWeight: '500' }}>🔍 Overview Mode</span>
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
                  <span style={{ fontWeight: '500' }}>🌐 Network Relationships</span>
                  <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                    Show connections between related models (same library, pipeline, or tags). Blue = library, Pink = pipeline.
                  </div>
                </div>
              </label>

              {showNetworkEdges && (
                <div style={{ marginLeft: '1.5rem', marginBottom: '1rem', padding: '0.75rem', background: 'white', borderRadius: '4px', border: '1px solid #d0d0d0' }}>
                  <label style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                    Connection Type
                  </label>
                  <select 
                    value={networkEdgeType} 
                    onChange={(e) => setNetworkEdgeType(e.target.value as 'library' | 'pipeline' | 'combined')}
                    style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid #d0d0d0', fontSize: '0.85rem' }}
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
                  <span style={{ fontWeight: '500' }}>📦 Structural Groupings</span>
                  <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                    Highlight clusters and groups with wireframe boundaries. Shows top library and pipeline clusters.
                  </div>
                </div>
              </label>
            </div>
          )}

          {/* Quick Filters */}
          <div className="sidebar-section">
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              ⚡ Quick Actions
            </h3>
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
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              🌳 Hierarchy Navigation
            </h3>
            <label style={{ marginBottom: '1rem', display: 'block' }}>
              <span style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>
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
                style={{ width: '100%', marginTop: '0.5rem' }}
              />
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem', display: 'flex', justifyContent: 'space-between' }}>
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
              <span style={{ fontSize: '0.9rem' }}>Show Distance Heatmap</span>
            </label>
            {selectedModel && (
              <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: '#f5f5f5', borderRadius: '4px', fontSize: '0.85rem' }}>
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
                    borderRadius: '2px',
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
                    borderRadius: '2px',
                    cursor: 'pointer',
                    marginBottom: '0.5rem'
                  }}
                >
                  Clear Path
                </button>
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <h3 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              color: '#5e35b1'
            }}>
              👥 Family Tree Explorer
            </h3>
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
                  showLabels={showLabels}
                  zoomLevel={zoomLevel}
                  nodeDensity={nodeDensity}
                  renderingStyle={renderingStyle}
                        showNetworkEdges={showNetworkEdges}
                        showStructuralGroups={showStructuralGroups}
                        overviewMode={overviewMode}
                        networkEdgeType={networkEdgeType}
                        onPointClick={(model) => {
                          setSelectedModel(model);
                          setIsModalOpen(true);
                        }}
                        selectedModelId={selectedModel?.model_id || familyTreeModelId}
                        selectedModel={selectedModel}
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
                        highlightedPath={highlightedPath}
                        showDistanceHeatmap={showDistanceHeatmap && !!selectedModel}
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
                      fontFamily: "'Instrument Sans', sans-serif"
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
                      onRegionSelect={(center: { x: number; y: number; z: number }) => {
                        setViewCenter(center);
                        // Camera will automatically animate to this position via targetViewCenter prop
                      }}
                        selectedModelId={selectedModel?.model_id || familyTreeModelId}
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
                        fontFamily: "'Instrument Sans', sans-serif"
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
              {viewMode === 'distribution' && (
                <DistributionView data={data} width={width} height={height} />
              )}
              {viewMode === 'stacked' && (
                <StackedView data={data} width={width} height={height} />
              )}
              {viewMode === 'heatmap' && (
                <HeatmapView data={data} width={width} height={height} />
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

