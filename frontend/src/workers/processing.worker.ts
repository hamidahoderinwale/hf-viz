/**
 * Enhanced Web Worker for heavy data processing.
 * Offloads filtering, sorting, and clustering to avoid blocking UI.
 */

import { ModelPoint } from '../types';

interface WorkerMessage {
  type: 'FILTER' | 'SORT' | 'SAMPLE' | 'CLUSTER';
  payload: any;
}

// Worker context
const ctx: Worker = self as any;

ctx.addEventListener('message', (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'FILTER':
        const filtered = filterData(payload.data, payload.filters);
        ctx.postMessage({ type: 'FILTER_RESULT', data: filtered });
        break;

      case 'SORT':
        const sorted = sortData(payload.data, payload.sortBy, payload.direction);
        ctx.postMessage({ type: 'SORT_RESULT', data: sorted });
        break;

      case 'SAMPLE':
        const sampled = sampleData(payload.data, payload.size, payload.strategy);
        ctx.postMessage({ type: 'SAMPLE_RESULT', data: sampled });
        break;

      case 'CLUSTER':
        const clusters = computeSimpleClusters(payload.data, payload.numClusters);
        ctx.postMessage({ type: 'CLUSTER_RESULT', data: clusters });
        break;

      default:
        ctx.postMessage({ type: 'ERROR', error: 'Unknown message type' });
    }
  } catch (error) {
    ctx.postMessage({ 
      type: 'ERROR', 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

function filterData(data: ModelPoint[], filters: any): ModelPoint[] {
  return data.filter(model => {
    if (filters.minDownloads && model.downloads < filters.minDownloads) return false;
    if (filters.minLikes && model.likes < filters.minLikes) return false;
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      if (!model.model_id.toLowerCase().includes(query)) return false;
    }
    if (filters.libraries && filters.libraries.length > 0) {
      if (!model.library_name || !filters.libraries.includes(model.library_name)) {
        return false;
      }
    }
    if (filters.pipelines && filters.pipelines.length > 0) {
      if (!model.pipeline_tag || !filters.pipelines.includes(model.pipeline_tag)) {
        return false;
      }
    }
    return true;
  });
}

function sortData(data: ModelPoint[], sortBy: keyof ModelPoint, direction: 'asc' | 'desc'): ModelPoint[] {
  const sorted = [...data].sort((a, b) => {
    const aVal = a[sortBy];
    const bVal = b[sortBy];
    
    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;
    
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return direction === 'asc' 
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }
    
    return direction === 'asc'
      ? (aVal as any) - (bVal as any)
      : (bVal as any) - (aVal as any);
  });
  
  return sorted;
}

function sampleData(data: ModelPoint[], size: number, strategy: 'random' | 'stratified'): ModelPoint[] {
  if (data.length <= size) return data;

  if (strategy === 'stratified' && data.length > 0) {
    // Stratified sampling by library
    const byLibrary = new Map<string, ModelPoint[]>();
    
    for (const model of data) {
      const lib = model.library_name || 'unknown';
      if (!byLibrary.has(lib)) {
        byLibrary.set(lib, []);
      }
      byLibrary.get(lib)!.push(model);
    }

    const sampled: ModelPoint[] = [];
    const libraries = Array.from(byLibrary.keys());
    const perLibrary = Math.floor(size / libraries.length);

    for (const lib of libraries) {
      const models = byLibrary.get(lib)!;
      const n = Math.min(perLibrary, models.length);
      
      // Simple random sample
      for (let i = 0; i < n; i++) {
        const idx = Math.floor(Math.random() * models.length);
        sampled.push(models[idx]);
        models.splice(idx, 1);
      }
    }

    // Fill remaining
    const remaining = size - sampled.length;
    if (remaining > 0) {
      const allRemaining = Array.from(byLibrary.values()).flat();
      for (let i = 0; i < remaining && allRemaining.length > 0; i++) {
        const idx = Math.floor(Math.random() * allRemaining.length);
        sampled.push(allRemaining[idx]);
        allRemaining.splice(idx, 1);
      }
    }

    return sampled;
  } else {
    // Random sampling
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, size);
  }
}

function computeSimpleClusters(data: ModelPoint[], numClusters: number): number[] {
  // Simple grid-based clustering for visualization
  const clusters: number[] = new Array(data.length);
  
  if (data.length === 0) return clusters;

  // Find bounds
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  
  for (const model of data) {
    minX = Math.min(minX, model.x);
    maxX = Math.max(maxX, model.x);
    minY = Math.min(minY, model.y);
    maxY = Math.max(maxY, model.y);
  }

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const gridSize = Math.ceil(Math.sqrt(numClusters));

  for (let i = 0; i < data.length; i++) {
    const model = data[i];
    const gridX = Math.floor(((model.x - minX) / rangeX) * gridSize);
    const gridY = Math.floor(((model.y - minY) / rangeY) * gridSize);
    clusters[i] = gridY * gridSize + gridX;
  }

  return clusters;
}

export {};



