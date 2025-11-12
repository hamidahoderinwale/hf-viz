/**
 * Main React app component using Visx for visualization.
 */
import React, { useState, useEffect, useCallback } from 'react';
import ScatterPlot from './components/ScatterPlot';
import ModelModal from './components/ModelModal';
import { ModelPoint, Stats } from './types';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [data, setData] = useState<ModelPoint[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelPoint | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Filters
  const [minDownloads, setMinDownloads] = useState(0);
  const [minLikes, setMinLikes] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [colorBy, setColorBy] = useState('library_name');
  const [sizeBy, setSizeBy] = useState('downloads');

  // Dimensions
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

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        min_downloads: minDownloads.toString(),
        min_likes: minLikes.toString(),
        color_by: colorBy,
        size_by: sizeBy,
        max_points: '5000',
      });
      if (searchQuery) {
        params.append('search_query', searchQuery);
      }

      const response = await fetch(`${API_BASE}/api/models?${params}`);
      if (!response.ok) throw new Error('Failed to fetch models');
      const models = await response.json();
      setData(models);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [minDownloads, minLikes, searchQuery, colorBy, sizeBy]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    // Fetch stats once
    fetch(`${API_BASE}/api/stats`)
      .then(res => res.json())
      .then(setStats)
      .catch(console.error);
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
            Read the full paper →
          </a>
        </p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>
          <strong>Authors:</strong> Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg
        </p>
        {stats && (
          <div className="stats">
            <span>Total Models: {stats.total_models.toLocaleString()}</span>
            <span>Libraries: {stats.unique_libraries}</span>
            <span>Pipelines: {stats.unique_pipelines}</span>
          </div>
        )}
      </header>

      <div className="main-content">
        <aside className="sidebar">
          <h2>Filters</h2>
          
          <label>
            Search:
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Model ID or tags..."
            />
          </label>

          <label>
            Min Downloads: {minDownloads.toLocaleString()}
            <input
              type="range"
              min="0"
              max="1000000"
              step="1000"
              value={minDownloads}
              onChange={(e) => setMinDownloads(Number(e.target.value))}
            />
          </label>

          <label>
            Min Likes: {minLikes.toLocaleString()}
            <input
              type="range"
              min="0"
              max="10000"
              step="10"
              value={minLikes}
              onChange={(e) => setMinLikes(Number(e.target.value))}
            />
          </label>

          <label>
            Color By:
            <select value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
              <option value="library_name">Library</option>
              <option value="pipeline_tag">Pipeline</option>
              <option value="downloads">Downloads</option>
              <option value="likes">Likes</option>
            </select>
          </label>

          <label>
            Size By:
            <select value={sizeBy} onChange={(e) => setSizeBy(e.target.value)}>
              <option value="downloads">Downloads</option>
              <option value="likes">Likes</option>
              <option value="trendingScore">Trending Score</option>
              <option value="none">None</option>
            </select>
          </label>
        </aside>

        <main className="visualization">
          {loading && <div className="loading">Loading models...</div>}
          {error && <div className="error">Error: {error}</div>}
          {!loading && !error && data.length === 0 && (
            <div className="empty">No models match the filters</div>
          )}
          {!loading && !error && data.length > 0 && (
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
            />
          )}
        </main>

        <ModelModal
          model={selectedModel}
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
        />
      </div>
    </div>
  );
}

export default App;

