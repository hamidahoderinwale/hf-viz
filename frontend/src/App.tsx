import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Palette, Maximize2, Eye } from 'lucide-react';
import IntroModal from './components/ui/IntroModal';
import ScatterPlot3D from './components/visualizations/ScatterPlot3D';
import NetworkGraph from './components/visualizations/NetworkGraph';
import DistributionView from './components/visualizations/DistributionView';
import type { Cluster } from './components/controls/ClusterFilter';
import ErrorBoundary from './components/ui/ErrorBoundary';
import LiveModelCounter from './components/ui/LiveModelCounter';
import ModelPopup from './components/ui/ModelPopup';
import AnalyticsPage from './pages/AnalyticsPage';
import FamiliesPage from './pages/FamiliesPage';
// Types & Utils
import { ModelPoint, Stats, SearchResult } from './types';
import IntegratedSearch from './components/controls/IntegratedSearch';
import cache, { IndexedDBCache } from './utils/data/indexedDB';
import { debounce } from './utils/debounce';
import requestManager from './utils/api/requestManager';
import { fetchWithMsgPack, decodeModelsMsgPack } from './utils/api/msgpackDecoder';
import { useFilterStore, ViewMode, ColorByOption, SizeByOption, FilterState } from './stores/filterStore';
import { API_BASE } from './config/api';
import './App.css';

const logger = {
  error: (message: string, error?: unknown) => {
    // Suppress NetworkError messages - they're expected during backend startup
    if (error instanceof Error) {
      const errorMsg = error.message.toLowerCase();
      if (errorMsg.includes('networkerror') || 
          errorMsg.includes('failed to fetch') ||
          errorMsg.includes('network request failed')) {
        // Silently ignore network errors during startup
        return;
      }
    }
    if (process.env.NODE_ENV === 'development') {
      console.error(message, error);
    }
  },
};

function App() {
  const {
    viewMode,
    colorBy,
    sizeBy,
    colorScheme,
    theme,
    selectedClusters,
    searchQuery,
    minDownloads,
    minLikes,
    setViewMode,
    setColorBy,
    setSizeBy,
    setColorScheme,
    setSearchQuery,
  } = useFilterStore();

  // Initialize theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const [data, setData] = useState<ModelPoint[]>([]);
  const [, setFilteredCount] = useState<number | null>(null);
  const [, setReturnedCount] = useState<number | null>(null);
  const [, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [, setLoadingMessage] = useState<string>('Loading models...');
  const [, setLoadingProgress] = useState<number | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelPoint | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [showIntro, setShowIntro] = useState(() => {
    // Check if user has dismissed the intro before
    return localStorage.getItem('hf-intro-dismissed') !== 'true';
  });
  const [baseModelsOnly] = useState(false);
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [semanticSimilarityMode] = useState(false);
  const [semanticQueryModel] = useState<string | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showFamilies, setShowFamilies] = useState(false);
  
  const [, setSearchResults] = useState<SearchResult[]>([]);
  const [searchInput] = useState('');
  const [, setShowSearchResults] = useState(false);
  const [projectionMethod] = useState<'umap' | 'tsne'>('umap');
  const [bookmarkedModels, setBookmarkedModels] = useState<string[]>([]);
  const [, setHoveredModel] = useState<ModelPoint | null>(null);
  const [, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [, setLiveModelCount] = useState<number | null>(null);
  
  const [useGraphEmbeddings] = useState(false);
  const [, setEmbeddingType] = useState<string>('text-only');
  const [, setClusters] = useState<Cluster[]>([]);
  const [, setClustersLoading] = useState(false);
  

  const [width, setWidth] = useState(window.innerWidth - 240); // Account for left sidebar
  const [height, setHeight] = useState(window.innerHeight - 160); // Account for header and top bar

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth - 240); // Account for left sidebar (240px)
      setHeight(window.innerHeight - 160); // Account for header (~80px) and top bar (~80px)
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Stable callbacks for ScatterPlot3D to prevent re-renders
  const handlePointClick = useCallback((model: ModelPoint) => {
    setSelectedModel(model);
    setIsModalOpen(true);
  }, []);

  const handleHover = useCallback((model: ModelPoint | null, position?: { x: number; y: number }) => {
    setHoveredModel(model);
    if (model && position) {
      setTooltipPosition(position);
    } else {
      setTooltipPosition(null);
    }
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
        // Don't recursively call fetchData - it causes infinite loops
        // The cache is valid, use it
        return;
      }
      let models: ModelPoint[];
      let count: number | null = null;
      
      if (semanticSimilarityMode && semanticQueryModel) {
        setLoadingMessage('Finding similar models...');
        setLoadingProgress(50);
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
        setLoadingMessage('Loading embeddings and coordinates...');
        setLoadingProgress(40);
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
        
        // Request up to 150k models for scatter plots, limit network graph for performance
        params.append('max_points', viewMode === 'network' ? '500' : '150000');
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
              setReturnedCount(result.returned_count ?? models.length);
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
            setReturnedCount(result.returned_count ?? models.length);
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
  }, [minDownloads, minLikes, searchQuery, projectionMethod, baseModelsOnly, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, selectedClusters, viewMode]);

  // Debounce times for different control types
  const SLIDER_DEBOUNCE_MS = 500; // Sliders need longer debounce
  const SEARCH_DEBOUNCE_MS = 300; // Search debounce
  const DROPDOWN_DEBOUNCE_MS = 200; // Dropdowns need shorter debounce

  const debouncedFetchData = useMemo(
    () => debounce(fetchData, SEARCH_DEBOUNCE_MS),
    [fetchData]
  );

  // Debounced setters for sliders (minDownloads, minLikes)
  // Debounced setter for search
  const debouncedSetSearchQuery = useMemo(
    () => debounce((query: string) => {
      setSearchQuery(query);
    }, SEARCH_DEBOUNCE_MS),
    [setSearchQuery]
  );

  // Debounced setters for dropdowns
  const debouncedSetColorBy = useMemo(
    () => debounce((value: ColorByOption) => {
      setColorBy(value);
    }, DROPDOWN_DEBOUNCE_MS),
    [setColorBy]
  );

  const debouncedSetSizeBy = useMemo(
    () => debounce((value: SizeByOption) => {
      setSizeBy(value);
    }, DROPDOWN_DEBOUNCE_MS),
    [setSizeBy]
  );

  const debouncedSetViewMode = useMemo(
    () => debounce((mode: ViewMode) => {
      setViewMode(mode);
    }, DROPDOWN_DEBOUNCE_MS),
    [setViewMode]
  );

  const debouncedSetColorScheme = useMemo(
    () => debounce((scheme: FilterState['colorScheme']) => {
      setColorScheme(scheme);
    }, DROPDOWN_DEBOUNCE_MS),
    [setColorScheme]
  );

  // Local state for search to show immediate feedback
  const [localSearchQuery, setLocalSearchQuery] = useState(searchQuery);

  // Sync local state with store state when store changes externally
  useEffect(() => {
    setLocalSearchQuery(searchQuery);
  }, [searchQuery]);

  // Cleanup debounced functions on unmount
  useEffect(() => {
    return () => {
      debouncedSetSearchQuery.cancel();
      debouncedSetColorBy.cancel();
      debouncedSetSizeBy.cancel();
      debouncedSetViewMode.cancel();
      debouncedSetColorScheme.cancel();
    };
  }, [debouncedSetSearchQuery, debouncedSetColorBy, debouncedSetSizeBy, debouncedSetViewMode, debouncedSetColorScheme]);

  // Initial fetch on mount (with delay to allow backend to start)
  useEffect(() => {
    // Delay initial fetch to allow backend to start
    const timer = setTimeout(() => {
      fetchData();
    }, 500); // 500ms delay
    
    return () => clearTimeout(timer);
  }, [fetchData]); // Include fetchData dependency

  // Consolidated effect to handle both search and filter changes
  // NOTE: colorBy and sizeBy are CLIENT-SIDE only - don't refetch data for these changes
  // Skip if this is the initial mount (first 600ms) - let the initial fetch effect handle it
  const hasMounted = useRef(false);
  useEffect(() => {
    if (!hasMounted.current) {
      hasMounted.current = true;
      return;
    }
    
    // For search queries or filter changes, use debounced version
    debouncedFetchData();
    return () => {
      debouncedFetchData.cancel();
    };
  }, [searchQuery, minDownloads, minLikes, baseModelsOnly, projectionMethod, semanticSimilarityMode, semanticQueryModel, useGraphEmbeddings, selectedClusters, viewMode, debouncedFetchData]);

  // Fetch live model count
  useEffect(() => {
    const fetchLiveCount = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/model-count/current?use_models_page=true&use_cache=true`);
        if (response.ok) {
          const data = await response.json();
          setLiveModelCount(data.total_models);
        }
      } catch (err) {
        // Silently fail - live count is optional
      }
    };
    
    fetchLiveCount();
    // Refresh every 5 minutes
    const interval = setInterval(fetchLiveCount, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchStats = async (retries = 3) => {
      const cacheKey = 'stats';
      const cachedStats = await cache.getCachedStats(cacheKey);
      
      // Always fetch fresh stats on initial load, ignore stale cache
      // This fixes the issue of showing old data (1000 models) when backend has 5000
      if (cachedStats) {
        // Show cached data temporarily
        setStats(cachedStats);
      }
      
      // Always fetch fresh stats to update with retry logic
      for (let i = 0; i < retries; i++) {
        try {
          const response = await fetch(`${API_BASE}/api/stats`);
          if (!response.ok) throw new Error('Failed to fetch stats');
          const statsData = await response.json();
          await cache.cacheStats(cacheKey, statsData);
          setStats(statsData);
          return; // Success
        } catch (err) {
          if (i === retries - 1) {
            // Only log on final retry failure (and not NetworkError)
            if (err instanceof Error && !err.message.includes('NetworkError')) {
              logger.error('Error fetching stats:', err);
            }
          } else {
            // Wait before retry (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
          }
        }
      }
    };

    // Delay initial fetch to allow backend to start
    const timer = setTimeout(() => {
      fetchStats();
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  // Fetch clusters with retry logic
  useEffect(() => {
    const fetchClusters = async (retries = 3) => {
      setClustersLoading(true);
      for (let i = 0; i < retries; i++) {
        try {
          const response = await fetch(`${API_BASE}/api/clusters`);
          if (!response.ok) throw new Error('Failed to fetch clusters');
          const data = await response.json();
          setClusters(data.clusters || []);
          setClustersLoading(false);
          return; // Success
        } catch (err) {
          if (i === retries - 1) {
            // Only log on final retry failure
            if (err instanceof Error && !err.message.includes('NetworkError')) {
              logger.error('Error fetching clusters:', err);
            }
            setClusters([]);
            setClustersLoading(false);
          } else {
            // Wait before retry (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
          }
        }
      }
    };

    // Delay initial fetch to allow backend to start
    const timer = setTimeout(() => {
      fetchClusters();
    }, 1000);
    
    return () => clearTimeout(timer);
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

  // Bookmark management
  const toggleBookmark = useCallback((modelId: string) => {
    setBookmarkedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  }, []);

  return (
    <ErrorBoundary>
      <div className="App">
      <div className="app-layout">
        {/* Left Navigation Sidebar */}
        <aside className={`nav-sidebar ${navCollapsed ? 'collapsed' : ''}`}>
          <div className="nav-sidebar-header">
            <h1>Hugging Face Model Lineage Viewer</h1>
            <button
              className="nav-collapse-toggle"
              onClick={() => setNavCollapsed(!navCollapsed)}
              aria-label={navCollapsed ? 'Expand navigation' : 'Collapse navigation'}
              title={navCollapsed ? 'Expand navigation' : 'Collapse navigation'}
            >
              {navCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
          </div>
          {!navCollapsed && (
            <nav className="nav-tabs">
              <button
                onClick={() => {
                  setShowAnalytics(false);
                  setShowFamilies(false);
                }}
                className={`nav-tab ${!showAnalytics && !showFamilies ? 'active' : ''}`}
                title="3D scatter plot of model embeddings — explore the model space interactively"
              >
                Visualization
              </button>
              <button
                onClick={() => {
                  setShowAnalytics(false);
                  setShowFamilies(true);
                }}
                className={`nav-tab ${showFamilies ? 'active' : ''}`}
                title="Browse model families and their lineage trees"
              >
                Families
              </button>
              <button
                onClick={() => {
                  setShowFamilies(false);
                  setShowAnalytics(true);
                }}
                className={`nav-tab ${showAnalytics ? 'active' : ''}`}
                title="Top models, trends, and statistics"
              >
                Analytics
              </button>
            </nav>
          )}
          {!navCollapsed && (
            <div className="nav-sidebar-footer">
              <div className="nav-links">
                <a href="https://arxiv.org/abs/2508.06811" target="_blank" rel="noopener noreferrer" title="Read the research paper on arXiv">Paper</a>
                <a href="https://github.com/bendlaufer/ai-ecosystem" target="_blank" rel="noopener noreferrer" title="View source code on GitHub">GitHub</a>
                <a href="https://huggingface.co/modelbiome" target="_blank" rel="noopener noreferrer" title="Access the dataset on Hugging Face">Dataset</a>
              </div>
            </div>
          )}
        </aside>

        {/* Main Content Area */}
        <div className="app-main">
          <div className="app-content">
            {showAnalytics ? (
              <AnalyticsPage />
            ) : showFamilies ? (
              <FamiliesPage />
            ) : (
      <div className="visualization-layout">
        <div className="control-bar">
          <div className="control-bar-content">
            {/* Left: Title (only shown when nav is collapsed) */}
            <div className="control-bar-left">
              {navCollapsed && (
                <span className="control-bar-title">HF Model Lineage Viewer</span>
              )}
            </div>

            {/* Center: Core Controls */}
            <div className="control-bar-center">

              {/* Color by */}
              <div className="control-group">
                <Palette size={14} className="control-icon" />
                <select 
                  value={colorBy} 
                  onChange={(e) => debouncedSetColorBy(e.target.value as ColorByOption)}
                  className="control-select"
                  title="Color points by attribute - Changes what determines each point's color"
                >
                  <option value="family_depth">Family Depth</option>
                  <option value="library_name">ML Library</option>
                  <option value="pipeline_tag">Task Type</option>
                  <option value="downloads">Downloads</option>
                  <option value="likes">Likes</option>
                </select>
                {(colorBy === 'downloads' || colorBy === 'likes' || colorBy === 'trending_score') && (
                  <select 
                    value={colorScheme} 
                    onChange={(e) => debouncedSetColorScheme(e.target.value as any)}
                    className="control-select control-select-small"
                    title="Color gradient style"
                  >
                    <option value="viridis">Viridis</option>
                    <option value="plasma">Plasma</option>
                    <option value="inferno">Inferno</option>
                    <option value="coolwarm">Cool-Warm</option>
                  </select>
                )}
              </div>

              <span className="control-divider" />

              {/* Size by */}
              <div className="control-group">
                <Maximize2 size={14} className="control-icon" />
                <select 
                  value={sizeBy} 
                  onChange={(e) => debouncedSetSizeBy(e.target.value as SizeByOption)}
                  className="control-select"
                  title="Size points by attribute - Larger values = bigger points"
                >
                  <option value="downloads">By Downloads</option>
                  <option value="likes">By Likes</option>
                  <option value="none">Uniform Size</option>
                </select>
              </div>

              <span className="control-divider" />

              {/* Stats summary */}
              <div className="control-stats" title="Number of models currently loaded and visible in the visualization">
                <Eye size={14} className="control-icon" />
                <span className="control-stats-text">
                  {data.length.toLocaleString()} models
                </span>
              </div>

            </div>

            {/* Right: Integrated Search */}
            <div className="control-bar-right">
              <IntegratedSearch
                value={localSearchQuery}
                onChange={(value) => {
                  setLocalSearchQuery(value);
                  debouncedSetSearchQuery(value);
                }}
                onSelect={(result) => {
                  const modelPoint: ModelPoint = {
                    model_id: result.model_id,
                    x: result.x || 0,
                    y: result.y || 0,
                    z: result.z || 0,
                    downloads: result.downloads || 0,
                    likes: result.likes || 0,
                    trending_score: null,
                    tags: null,
                    licenses: null,
                    cluster_id: null,
                    created_at: null,
                    library_name: result.library_name || null,
                    pipeline_tag: result.pipeline_tag || null,
                    parent_model: null,
                    family_depth: result.family_depth || null,
                  };
                  setSelectedModel(modelPoint);
                  setIsModalOpen(true);
                  setLocalSearchQuery('');
                  setSearchQuery('');
                }}
                onZoomTo={(x, y, z) => {
                  // Zoom to point - reserved for future implementation
                }}
              />
            </div>
          </div>
        </div>


        <main className="visualization">
          {/* Intro Modal */}
          {showIntro && !loading && data.length > 0 && (
            <IntroModal onClose={() => setShowIntro(false)} />
          )}
          
          {loading && <div className="loading">Loading models...</div>}
          {error && <div className="error">Error: {error}</div>}
          {!loading && !error && data.length === 0 && (
            <div className="empty">No models match the filters</div>
          )}
          {!loading && !error && data.length > 0 && (
            <>
              {viewMode === '3d' && (
                <>
                  <ScatterPlot3D
                    data={data}
                    colorBy={colorBy}
                    sizeBy={sizeBy}
                    colorScheme={colorScheme}
                    hoveredModel={null}
                    onPointClick={handlePointClick}
                    onHover={handleHover}
                  />
                </>
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
          
          {/* Live Model Counter - Bottom Left */}
          <LiveModelCounter
            pollInterval={60000}
            showGrowth={true}
          />
          
          {/* Model Popup - Bottom Left */}
          <ModelPopup
            model={selectedModel}
            isOpen={isModalOpen}
            onClose={() => setIsModalOpen(false)}
            onBookmark={selectedModel ? () => toggleBookmark(selectedModel.model_id) : undefined}
            isBookmarked={selectedModel ? bookmarkedModels.includes(selectedModel.model_id) : false}
          />
        </main>
            </div>
          )}
          </div>
        </div>
      </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;

