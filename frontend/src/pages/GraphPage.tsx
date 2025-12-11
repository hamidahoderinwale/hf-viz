import React, { useState, useEffect, useCallback } from 'react';
import ForceDirectedGraph, { EdgeType, GraphNode } from '../components/visualizations/ForceDirectedGraph';
import ForceDirectedGraph3D from '../components/visualizations/ForceDirectedGraph3D';
import ScatterPlot3D from '../components/visualizations/ScatterPlot3D';
import { fetchFamilyNetwork, fetchFullDerivativeNetwork, getAvailableEdgeTypes } from '../utils/api/graphApi';
import LoadingProgress from '../components/ui/LoadingProgress';
import { ModelPoint } from '../types';
// Simple search input for graph page
import { API_BASE } from '../config/api';
import './GraphPage.css';

const ALL_EDGE_TYPES: EdgeType[] = ['finetune', 'quantized', 'adapter', 'merge', 'parent'];

type ViewMode = 'graph' | 'embedding' | 'graph3d';
type GraphMode = 'family' | 'full';

export default function GraphPage() {
  const [modelId, setModelId] = useState<string>('');
  const [viewMode, setViewMode] = useState<ViewMode>('graph3d');
  const [graphMode, setGraphMode] = useState<GraphMode>('full');
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<any[]>([]);
  const [embeddingData, setEmbeddingData] = useState<ModelPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingEmbedding, setLoadingEmbedding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelPoint | null>(null);
  const [enabledEdgeTypes, setEnabledEdgeTypes] = useState<Set<EdgeType>>(
    new Set(ALL_EDGE_TYPES)
  );
  const [maxDepth, setMaxDepth] = useState<number | undefined>(5);
  const [graphStats, setGraphStats] = useState<any>(null);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [colorBy, setColorBy] = useState<string>('library_name');
  const [sizeBy, setSizeBy] = useState<string>('downloads');

  // Load graph when modelId, maxDepth, or graphMode changes
  // For full mode, load immediately on mount and whenever graphMode is 'full'
  useEffect(() => {
    const loadGraph = async () => {
      setLoading(true);
      setError(null);
      try {
        let data;
        
        if (graphMode === 'full') {
          // Load full derivative network for all models - no search needed
          data = await fetchFullDerivativeNetwork({
            edgeTypes: undefined, // Get all types, filter client-side
            includeEdgeAttributes: true,
          });
        } else {
          // Load family network for specific model - requires modelId
          if (!modelId.trim()) {
            setNodes([]);
            setLinks([]);
            setEmbeddingData([]);
            setGraphStats(null);
            setLoading(false);
            return;
          }
          
          data = await fetchFamilyNetwork(modelId, {
            maxDepth,
            edgeTypes: undefined, // Get all types, filter client-side
            includeEdgeAttributes: true,
          });
        }

        setNodes(data.nodes || []);
        setLinks(data.links || []);
        setGraphStats(data.statistics);

        // Update enabled edge types based on available types (only on first load)
        if (data.links && data.links.length > 0) {
          const availableTypes = getAvailableEdgeTypes(data.links);
          // If no types are currently enabled, enable all available
          if (enabledEdgeTypes.size === 0 && availableTypes.size > 0) {
            setEnabledEdgeTypes(new Set(availableTypes));
          }
        }
      } catch (err: any) {
        const errorMessage = err.message || 'Failed to load graph';
        console.error('Error loading graph:', err);
        setError(errorMessage);
        setNodes([]);
        setLinks([]);
        setEmbeddingData([]);
      } finally {
        setLoading(false);
      }
    };

    // Always load graph - for full mode it loads immediately, for family mode it waits for modelId
    loadGraph();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, maxDepth, graphMode]); // Reload when modelId, maxDepth, or graphMode changes

  // Load embedding data when switching to embedding view
  // This loads ALL models for embedding space view, independent of graph mode
  useEffect(() => {
    if (viewMode !== 'embedding') {
      setEmbeddingData([]);
      return;
    }

    const loadEmbeddingData = async () => {
      setLoadingEmbedding(true);
      try {
        // For embedding space, always load all models (or a large sample)
        // This is independent of whether we're in full network or family mode
        const params = new URLSearchParams({
          max_points: '150000', // Load up to 150k models for embedding view
          format: 'json',
        });
        
        const response = await fetch(`${API_BASE}/api/models?${params}`);
        if (!response.ok) throw new Error('Failed to fetch embedding data');
        
        const data = await response.json();
        const allModels: ModelPoint[] = Array.isArray(data) ? data : (data.models || []);
        
        setEmbeddingData(allModels);
      } catch (err: any) {
        console.error('Failed to load embedding data:', err);
        setEmbeddingData([]);
      } finally {
        setLoadingEmbedding(false);
      }
    };

    loadEmbeddingData();
  }, [viewMode]);

  // Handle search
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/search?q=${encodeURIComponent(query)}&limit=10`
      );
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      const results = Array.isArray(data) ? data : (data.models || []);
      setSearchResults(results);
      setShowSearchResults(true);
    } catch (err) {
      setSearchResults([]);
      setShowSearchResults(false);
    }
  }, []);

  const handleSearchResultClick = useCallback((model: any) => {
    setModelId(model.model_id);
    setShowSearchResults(false);
  }, []);

  const toggleEdgeType = useCallback((type: EdgeType) => {
    setEnabledEdgeTypes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  }, []);

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNodeId(node.id);
    setModelId(node.id);
    // Find corresponding model in embedding data if available
    if (embeddingData.length > 0) {
      const model = embeddingData.find(m => m.model_id === node.id);
      if (model) {
        setSelectedModel(model);
      }
    }
  }, [embeddingData]);

  const handleEmbeddingPointClick = useCallback((model: ModelPoint) => {
    setSelectedModel(model);
    setSelectedNodeId(model.model_id);
    setModelId(model.model_id);
    // When clicking a model in embedding space, switch to its lineage graph
    setGraphMode('family');
    setViewMode('graph3d'); // Show lineage in 3D graph view
  }, []);

  const containerRef = React.useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1000, height: 600 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: Math.max(600, Math.min(rect.height, 800)), // Cap at 800px max
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    // Also update on viewport changes
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => {
      window.removeEventListener('resize', updateDimensions);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div className="graph-page">
      <div className="page-header">
        <h1>Model Relationship Graph</h1>
        <p className="page-description">
          <strong>Full Network (Default):</strong> Interactive 3D force-directed graph showing ALL models and their derivative relationships.
          <br />
          <strong>Family Tree:</strong> Search for a specific model to view its lineage graph.
          <br />
          <strong>Embedding Space:</strong> View models in semantic space. Click any model to see its lineage graph.
          <br />
          Explore how models are connected through fine-tuning, quantization, adapters, and merges.
        </p>
      </div>

      <div className="graph-controls-panel">
        {graphMode === 'family' && (
          <div className="search-section">
            <input
              type="text"
              className="graph-search-input"
              placeholder="Search for a model to view its lineage..."
              value={modelId}
              onChange={(e) => {
                const value = e.target.value;
                setModelId(value);
                if (value.trim()) {
                  handleSearch(value);
                } else {
                  setSearchResults([]);
                  setShowSearchResults(false);
                }
              }}
              onFocus={() => {
                if (searchResults.length > 0) {
                  setShowSearchResults(true);
                }
              }}
            />
            {showSearchResults && searchResults.length > 0 && (
              <div className="search-results-dropdown">
                {searchResults.map((model) => (
                  <div
                    key={model.model_id}
                    className="search-result-item"
                    onClick={() => handleSearchResultClick(model)}
                  >
                    <div className="result-title">{model.model_id}</div>
                    <div className="result-meta">
                      {model.library_name && <span>{model.library_name}</span>}
                      {model.downloads > 0 && (
                        <span>{model.downloads.toLocaleString()} downloads</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        
        {graphMode === 'full' && (
          <div className="info-banner">
            <span>📊 Viewing full network: All models and relationships loaded</span>
          </div>
        )}

        <div className="graph-settings">
          <div className="setting-group">
            <label>Graph Mode:</label>
            <select
              value={graphMode}
              onChange={(e) => {
                setGraphMode(e.target.value as GraphMode);
                if (e.target.value === 'full') {
                  setViewMode('graph3d'); // Default to 3D for full graph
                  setModelId(''); // Clear model selection for full network
                }
              }}
              className="graph-mode-select"
            >
              <option value="full">Full Network (All Models)</option>
              <option value="family">Family Tree (Single Model Lineage)</option>
            </select>
          </div>

          <div className="setting-group">
            <label>View Mode:</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as ViewMode)}
              className="view-mode-select"
            >
              <option value="graph">Force-Directed Graph (2D)</option>
              <option value="graph3d">Force-Directed Graph (3D)</option>
              <option value="embedding">Embedding Space (3D)</option>
            </select>
          </div>

          {viewMode === 'embedding' && (
            <>
              <div className="setting-group">
                <label>Color By:</label>
                <select
                  value={colorBy}
                  onChange={(e) => setColorBy(e.target.value)}
                  className="color-by-select"
                >
                  <option value="library_name">Library</option>
                  <option value="pipeline_tag">Task Type</option>
                  <option value="downloads">Downloads</option>
                  <option value="likes">Likes</option>
                  <option value="family_depth">Family Depth</option>
                </select>
              </div>

              <div className="setting-group">
                <label>Size By:</label>
                <select
                  value={sizeBy}
                  onChange={(e) => setSizeBy(e.target.value)}
                  className="size-by-select"
                >
                  <option value="downloads">Downloads</option>
                  <option value="likes">Likes</option>
                  <option value="none">Uniform</option>
                </select>
              </div>
            </>
          )}

          {graphMode === 'family' && (
            <>
              <div className="setting-group">
                <label>Max Depth:</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={maxDepth || ''}
                  onChange={(e) => setMaxDepth(e.target.value ? parseInt(e.target.value) : undefined)}
                  className="depth-input"
                />
              </div>
              <div className="setting-group">
                <label>Current Model:</label>
                <div className="current-model">{modelId || 'None selected'}</div>
              </div>
            </>
          )}
          
          {graphMode === 'full' && graphStats && (
            <div className="setting-group">
              <label>Network Stats:</label>
              <div className="network-stats-summary">
                {graphStats.nodes?.toLocaleString()} models, {graphStats.edges?.toLocaleString()} relationships
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="graph-container" ref={containerRef}>
        {loading ? (
          <LoadingProgress message="Loading graph..." progress={0} />
        ) : error ? (
          <div className="graph-error">
            <p>Error: {error}</p>
            {error.includes('not found') && (
              <p className="error-hint">Try searching for a different model.</p>
            )}
          </div>
        ) : nodes.length === 0 ? (
          <div className="graph-empty">
            {graphMode === 'full' ? (
              <>
                <p>Loading full derivative network for all models...</p>
                <p className="empty-hint">This may take a moment as we load all models and their relationships.</p>
              </>
            ) : (
              <>
                <p>Search for a model above to visualize its lineage graph.</p>
                <p className="empty-hint">
                  Or switch to "Full Network" mode to see all models, or "Embedding Space" to browse and click models.
                  <br />
                  Try popular models like: <code>bert-base-uncased</code>, <code>gpt2</code>, or <code>t5-base</code>
                </p>
              </>
            )}
          </div>
        ) : viewMode === 'graph' ? (
          <>
            <ForceDirectedGraph
              width={dimensions.width}
              height={dimensions.height}
              nodes={nodes}
              links={links}
              onNodeClick={handleNodeClick}
              selectedNodeId={selectedNodeId}
              enabledEdgeTypes={enabledEdgeTypes}
              showLabels={true}
            />
            <EdgeTypeLegend
              edgeTypes={ALL_EDGE_TYPES}
              enabledTypes={enabledEdgeTypes}
              onToggle={toggleEdgeType}
            />
            {graphStats && (
              <div className="graph-stats">
                <div className="stat-item">
                  <span className="stat-label">Nodes:</span>
                  <span className="stat-value">{graphStats.nodes || nodes.length}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Edges:</span>
                  <span className="stat-value">{graphStats.edges || links.length}</span>
                </div>
                {graphStats.avg_degree && (
                  <div className="stat-item">
                    <span className="stat-label">Avg Degree:</span>
                    <span className="stat-value">{graphStats.avg_degree.toFixed(2)}</span>
                  </div>
                )}
              </div>
            )}
          </>
        ) : viewMode === 'graph3d' ? (
          <>
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
              <ForceDirectedGraph3D
                width={dimensions.width}
                height={dimensions.height}
                nodes={nodes}
                links={links}
                onNodeClick={handleNodeClick}
                selectedNodeId={selectedNodeId}
                enabledEdgeTypes={enabledEdgeTypes}
                showLabels={true}
              />
            </div>
            <EdgeTypeLegend
              edgeTypes={ALL_EDGE_TYPES}
              enabledTypes={enabledEdgeTypes}
              onToggle={toggleEdgeType}
            />
            {graphStats && (
              <div className="graph-stats">
                <div className="stat-item">
                  <span className="stat-label">Nodes:</span>
                  <span className="stat-value">{graphStats.nodes || nodes.length}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Edges:</span>
                  <span className="stat-value">{graphStats.edges || links.length}</span>
                </div>
                {graphStats.avg_degree && (
                  <div className="stat-item">
                    <span className="stat-label">Avg Degree:</span>
                    <span className="stat-value">{graphStats.avg_degree.toFixed(2)}</span>
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <>
            {loadingEmbedding ? (
              <LoadingProgress message="Loading embedding data for all models..." progress={0} />
            ) : embeddingData.length === 0 ? (
              <div className="graph-empty">
                <p>Loading embedding data...</p>
                <p className="empty-hint">This may take a moment for the full dataset.</p>
              </div>
            ) : (
              <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <ScatterPlot3D
                  data={embeddingData}
                  colorBy={colorBy}
                  sizeBy={sizeBy}
                  colorScheme="viridis"
                  onPointClick={handleEmbeddingPointClick}
                  hoveredModel={selectedModel}
                />
                <div className="embedding-info">
                  <div className="info-item">
                    <span className="info-label">Models:</span>
                    <span className="info-value">{embeddingData.length.toLocaleString()}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">View:</span>
                    <span className="info-value">Embedding Space</span>
                  </div>
                  {selectedModel && (
                    <div className="info-item">
                      <span className="info-label">Selected:</span>
                      <span className="info-value">{selectedModel.model_id}</span>
                    </div>
                  )}
                  <div className="info-hint">
                    💡 Click any model to view its lineage graph
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

interface EdgeTypeLegendProps {
  edgeTypes: EdgeType[];
  enabledTypes: Set<EdgeType>;
  onToggle: (type: EdgeType) => void;
}

const EDGE_COLORS: Record<EdgeType, string> = {
  finetune: '#3b82f6',
  quantized: '#10b981',
  adapter: '#f59e0b',
  merge: '#8b5cf6',
  parent: '#6b7280',
};

const EDGE_LABELS: Record<EdgeType, string> = {
  finetune: 'Fine-tuned',
  quantized: 'Quantized',
  adapter: 'Adapter',
  merge: 'Merged',
  parent: 'Parent',
};

function EdgeTypeLegend({ edgeTypes, enabledTypes, onToggle }: EdgeTypeLegendProps) {
  return (
    <div className="edge-type-legend">
      <h4>Relationship Types</h4>
      {edgeTypes.map((type) => (
        <div
          key={type}
          className={`edge-type-item ${!enabledTypes.has(type) ? 'disabled' : ''}`}
          onClick={() => onToggle(type)}
        >
          <div
            className="edge-type-color"
            style={{ backgroundColor: EDGE_COLORS[type] }}
          />
          <span className="edge-type-label">{EDGE_LABELS[type]}</span>
        </div>
      ))}
    </div>
  );
}
