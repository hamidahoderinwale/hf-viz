/**
 * Web Worker for efficient distance calculations in network graph.
 * Offloads CPU-intensive similarity calculations from main thread.
 */

interface ModelPoint {
  model_id: string;
  x: number;
  y: number;
  z: number;
}

interface DistanceCalculationMessage {
  type: 'calculateDistances';
  data: ModelPoint[];
  threshold: number;
  maxLinks: number;
  maxNodes?: number;
}

interface DistanceResult {
  links: Array<{ source: string; target: string; distance: number }>;
}

// Spatial grid for efficient nearest neighbor search
class SpatialGrid {
  private grid: Map<string, ModelPoint[]>;
  private cellSize: number;

  constructor(data: ModelPoint[], cellSize: number = 0.15) {
    this.grid = new Map();
    this.cellSize = cellSize;

    // Build spatial grid
    data.forEach((point) => {
      const cellKey = this.getCellKey(point.x, point.y);
      if (!this.grid.has(cellKey)) {
        this.grid.set(cellKey, []);
      }
      this.grid.get(cellKey)!.push(point);
    });
  }

  private getCellKey(x: number, y: number): string {
    const cellX = Math.floor(x / this.cellSize);
    const cellY = Math.floor(y / this.cellSize);
    return `${cellX},${cellY}`;
  }

  findNeighbors(point: ModelPoint, threshold: number): ModelPoint[] {
    const neighbors: ModelPoint[] = [];
    const cellX = Math.floor(point.x / this.cellSize);
    const cellY = Math.floor(point.y / this.cellSize);

    // Check current cell and 8 surrounding cells
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cellKey = `${cellX + dx},${cellY + dy}`;
        const cellPoints = this.grid.get(cellKey);
        if (cellPoints) {
          cellPoints.forEach((p) => {
            if (p.model_id !== point.model_id) {
              const dx2 = p.x - point.x;
              const dy2 = p.y - point.y;
              const distance = Math.sqrt(dx2 * dx2 + dy2 * dy2);
              if (distance < threshold) {
                neighbors.push({ ...p, distance } as any);
              }
            }
          });
        }
      }
    }

    return neighbors;
  }
}

function calculateDistances(
  data: ModelPoint[],
  threshold: number,
  maxLinks: number,
  maxNodes?: number
): DistanceResult {
  const links: Array<{ source: string; target: string; distance: number }> = [];
  const linkSet = new Set<string>();

  // Limit nodes for performance if specified
  const nodesToProcess = maxNodes ? data.slice(0, maxNodes) : data;

  // Build spatial grid for efficient neighbor search
  const spatialGrid = new SpatialGrid(data, threshold * 2);

  // Process each node
  nodesToProcess.forEach((model) => {
    if (links.length >= maxLinks) return;

    // Find neighbors using spatial grid (much faster than O(n²))
    const neighbors = spatialGrid.findNeighbors(model, threshold);

    // Sort by distance and take top 2
    neighbors
      .sort((a, b) => (a as any).distance - (b as any).distance)
      .slice(0, 2)
      .forEach((neighbor) => {
        const linkKey = [model.model_id, neighbor.model_id].sort().join('-');
        if (!linkSet.has(linkKey) && links.length < maxLinks) {
          linkSet.add(linkKey);
          links.push({
            source: model.model_id,
            target: neighbor.model_id,
            distance: (neighbor as any).distance,
          });
        }
      });
  });

  return { links };
}

// Handle messages from main thread
self.onmessage = (e: MessageEvent<DistanceCalculationMessage>) => {
  const { type, data, threshold, maxLinks, maxNodes } = e.data;

  if (type === 'calculateDistances') {
    try {
      const result = calculateDistances(data, threshold, maxLinks, maxNodes);
      self.postMessage({ type: 'result', result });
    } catch (error) {
      self.postMessage({
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
};

