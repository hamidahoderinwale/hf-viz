/**
 * Spatial sparsity utilities for reducing point density and improving navigability.
 * Filters points to ensure minimum distance between them.
 */

import { ModelPoint } from '../../types';

/**
 * Filter points to ensure minimum distance between them (spatial sparsification)
 * Uses a grid-based approach for efficiency
 */
export function applySpatialSparsity(
  points: ModelPoint[],
  minDistance: number,
  preserveImportant: Set<string> = new Set()
): ModelPoint[] {
  if (points.length === 0 || minDistance <= 0) return points;

  // Create a grid for efficient spatial queries
  const gridSize = minDistance;
  const grid = new Map<string, ModelPoint[]>();

  // Helper to get grid cell key
  const getGridKey = (x: number, y: number, z: number): string => {
    const gx = Math.floor(x / gridSize);
    const gy = Math.floor(y / gridSize);
    const gz = Math.floor(z / gridSize);
    return `${gx},${gy},${gz}`;
  };

  // Separate important and regular points
  const important: ModelPoint[] = [];
  const regular: ModelPoint[] = [];

  for (const point of points) {
    if (preserveImportant.has(point.model_id)) {
      important.push(point);
    } else {
      regular.push(point);
    }
  }

  // Always keep important points
  const result: ModelPoint[] = [...important];

  // Add important points to grid
  for (const point of important) {
    const key = getGridKey(point.x, point.y, point.z);
    if (!grid.has(key)) {
      grid.set(key, []);
    }
    grid.get(key)!.push(point);
  }

  // Process regular points with sparsity filtering
  for (const point of regular) {
    const key = getGridKey(point.x, point.y, point.z);
    
    // Check neighboring cells (3x3x3 = 27 cells)
    let tooClose = false;
    const [gx, gy, gz] = key.split(',').map(Number);
    
    for (let dx = -1; dx <= 1 && !tooClose; dx++) {
      for (let dy = -1; dy <= 1 && !tooClose; dy++) {
        for (let dz = -1; dz <= 1 && !tooClose; dz++) {
          const neighborKey = `${gx + dx},${gy + dy},${gz + dz}`;
          const neighbors = grid.get(neighborKey);
          
          if (neighbors) {
            for (const neighbor of neighbors) {
              const distance = Math.sqrt(
                Math.pow(point.x - neighbor.x, 2) +
                Math.pow(point.y - neighbor.y, 2) +
                Math.pow(point.z - neighbor.z, 2)
              );
              
              if (distance < minDistance) {
                tooClose = true;
                break;
              }
            }
          }
        }
      }
    }
    
    // If not too close to any existing point, add it
    if (!tooClose) {
      result.push(point);
      if (!grid.has(key)) {
        grid.set(key, []);
      }
      grid.get(key)!.push(point);
    }
  }

  return result;
}

/**
 * Apply adaptive sparsity based on dataset size
 * Larger datasets get more aggressive sparsity for better navigability
 */
export function getAdaptiveSparsityFactor(dataSize: number): number {
  if (dataSize < 5000) return 0; // No sparsity for very small datasets
  if (dataSize < 20000) return 0.03; // 3% of average distance
  if (dataSize < 50000) return 0.05; // 5% of average distance
  if (dataSize < 100000) return 0.08; // 8% of average distance
  if (dataSize < 200000) return 0.12; // 12% of average distance
  if (dataSize < 500000) return 0.18; // 18% of average distance
  return 0.25; // 25% for very large datasets (more sparse)
}

/**
 * Calculate average distance between points for adaptive sparsity
 */
export function calculateAverageDistance(points: ModelPoint[]): number {
  if (points.length < 2) return 0;
  
  // Sample a subset for efficiency
  const sampleSize = Math.min(1000, points.length);
  const step = Math.floor(points.length / sampleSize);
  const sample = [];
  
  for (let i = 0; i < points.length; i += step) {
    sample.push(points[i]);
    if (sample.length >= sampleSize) break;
  }
  
  let totalDistance = 0;
  let count = 0;
  
  // Calculate distances between sample points
  for (let i = 0; i < sample.length; i++) {
    for (let j = i + 1; j < Math.min(i + 10, sample.length); j++) {
      const dx = sample[i].x - sample[j].x;
      const dy = sample[i].y - sample[j].y;
      const dz = sample[i].z - sample[j].z;
      totalDistance += Math.sqrt(dx * dx + dy * dy + dz * dz);
      count++;
    }
  }
  
  return count > 0 ? totalDistance / count : 0;
}

