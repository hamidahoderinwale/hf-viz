/**
 * Spatial indexing utilities for efficient point queries in 3D space.
 * Uses Octree for O(log n) nearest neighbor and range queries.
 */

import { ModelPoint } from '../types';

interface BoundingBox {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

interface OctreeNode {
  bounds: BoundingBox;
  points: ModelPoint[];
  children: OctreeNode[] | null;
  depth: number;
}

const MAX_POINTS_PER_NODE = 100;
const MAX_DEPTH = 8;

/**
 * Calculate bounding box for a set of points
 */
function calculateBounds(points: ModelPoint[]): BoundingBox {
  if (points.length === 0) {
    return { minX: 0, minY: 0, minZ: 0, maxX: 0, maxY: 0, maxZ: 0 };
  }

  let minX = points[0].x;
  let minY = points[0].y;
  let minZ = points[0].z;
  let maxX = points[0].x;
  let maxY = points[0].y;
  let maxZ = points[0].z;

  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    minZ = Math.min(minZ, point.z);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
    maxZ = Math.max(maxZ, point.z);
  }

  return { minX, minY, minZ, maxX, maxY, maxZ };
}

/**
 * Check if a point is within bounding box
 */
function pointInBounds(point: ModelPoint, bounds: BoundingBox): boolean {
  return (
    point.x >= bounds.minX &&
    point.x <= bounds.maxX &&
    point.y >= bounds.minY &&
    point.y <= bounds.maxY &&
    point.z >= bounds.minZ &&
    point.z <= bounds.maxZ
  );
}

/**
 * Split bounding box into 8 octants
 */
function splitBounds(bounds: BoundingBox): BoundingBox[] {
  const midX = (bounds.minX + bounds.maxX) / 2;
  const midY = (bounds.minY + bounds.maxY) / 2;
  const midZ = (bounds.minZ + bounds.maxZ) / 2;

  return [
    // Front-top-left
    { minX: bounds.minX, maxX: midX, minY: midY, maxY: bounds.maxY, minZ: bounds.minZ, maxZ: midZ },
    // Front-top-right
    { minX: midX, maxX: bounds.maxX, minY: midY, maxY: bounds.maxY, minZ: bounds.minZ, maxZ: midZ },
    // Front-bottom-left
    { minX: bounds.minX, maxX: midX, minY: bounds.minY, maxY: midY, minZ: bounds.minZ, maxZ: midZ },
    // Front-bottom-right
    { minX: midX, maxX: bounds.maxX, minY: bounds.minY, maxY: midY, minZ: bounds.minZ, maxZ: midZ },
    // Back-top-left
    { minX: bounds.minX, maxX: midX, minY: midY, maxY: bounds.maxY, minZ: midZ, maxZ: bounds.maxZ },
    // Back-top-right
    { minX: midX, maxX: bounds.maxX, minY: midY, maxY: bounds.maxY, minZ: midZ, maxZ: bounds.maxZ },
    // Back-bottom-left
    { minX: bounds.minX, maxX: midX, minY: bounds.minY, maxY: midY, minZ: midZ, maxZ: bounds.maxZ },
    // Back-bottom-right
    { minX: midX, maxX: bounds.maxX, minY: bounds.minY, maxY: midY, minZ: midZ, maxZ: bounds.maxZ },
  ];
}

/**
 * Build octree node recursively
 */
function buildNode(points: ModelPoint[], bounds: BoundingBox, depth: number): OctreeNode {
  const node: OctreeNode = {
    bounds,
    points: [],
    children: null,
    depth,
  };

  // Filter points that are actually in this node's bounds
  const pointsInBounds = points.filter(p => pointInBounds(p, bounds));

  // If we have few points or reached max depth, store points in this node
  if (pointsInBounds.length <= MAX_POINTS_PER_NODE || depth >= MAX_DEPTH) {
    node.points = pointsInBounds;
    return node;
  }

  // Otherwise, split into children
  const childBounds = splitBounds(bounds);
  node.children = childBounds.map(childBound => buildNode(pointsInBounds, childBound, depth + 1));
  return node;
}

/**
 * Query points within a bounding box
 */
function queryRange(node: OctreeNode, bounds: BoundingBox, results: ModelPoint[]): void {
  // Check if node bounds intersect query bounds
  if (
    node.bounds.maxX < bounds.minX ||
    node.bounds.minX > bounds.maxX ||
    node.bounds.maxY < bounds.minY ||
    node.bounds.minY > bounds.maxY ||
    node.bounds.maxZ < bounds.minZ ||
    node.bounds.minZ > bounds.maxZ
  ) {
    return; // No intersection
  }

  // If this is a leaf node, check all points
  if (node.children === null) {
    for (const point of node.points) {
      if (pointInBounds(point, bounds)) {
        results.push(point);
      }
    }
    return;
  }

  // Otherwise, query children
  for (const child of node.children) {
    queryRange(child, bounds, results);
  }
}

/**
 * Find nearest neighbors to a point
 */
function findNearest(
  node: OctreeNode,
  point: ModelPoint,
  k: number,
  results: Array<{ point: ModelPoint; distance: number }>
): void {
  const distance = (p1: ModelPoint, p2: ModelPoint) => {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  // If leaf node, check all points
  if (node.children === null) {
    for (const p of node.points) {
      const dist = distance(point, p);
      results.push({ point: p, distance: dist });
    }
    // Sort by distance and keep only k nearest
    results.sort((a, b) => a.distance - b.distance);
    if (results.length > k) {
      results.splice(k);
    }
    return;
  }

  // Find which child contains the point
  const childBounds = splitBounds(node.bounds);
  for (let i = 0; i < node.children.length; i++) {
    if (pointInBounds(point, childBounds[i])) {
      findNearest(node.children[i], point, k, results);
      break;
    }
  }
}

/**
 * Spatial index class for efficient 3D point queries
 */
export class SpatialIndex {
  private root: OctreeNode | null = null;
  private allPoints: ModelPoint[] = [];

  constructor(points: ModelPoint[]) {
    this.allPoints = points;
    if (points.length > 0) {
      const bounds = calculateBounds(points);
      this.root = buildNode(points, bounds, 0);
    }
  }

  /**
   * Query all points within a bounding box
   */
  queryRange(bounds: BoundingBox): ModelPoint[] {
    if (!this.root) return [];
    const results: ModelPoint[] = [];
    queryRange(this.root, bounds, results);
    return results;
  }

  /**
   * Find k nearest neighbors to a point
   */
  findNearest(point: ModelPoint, k: number = 10): ModelPoint[] {
    if (!this.root) return [];
    const results: Array<{ point: ModelPoint; distance: number }> = [];
    findNearest(this.root, point, k, results);
    return results.map(r => r.point);
  }

  /**
   * Query points within a sphere (for frustum culling approximation)
   */
  querySphere(center: ModelPoint, radius: number): ModelPoint[] {
    const bounds: BoundingBox = {
      minX: center.x - radius,
      maxX: center.x + radius,
      minY: center.y - radius,
      maxY: center.y + radius,
      minZ: center.z - radius,
      maxZ: center.z + radius,
    };
    const candidates = this.queryRange(bounds);
    // Filter to actual sphere
    return candidates.filter(p => {
      const dx = p.x - center.x;
      const dy = p.y - center.y;
      const dz = p.z - center.z;
      return dx * dx + dy * dy + dz * dz <= radius * radius;
    });
  }

  /**
   * Get all points (for fallback)
   */
  getAllPoints(): ModelPoint[] {
    return this.allPoints;
  }
}

/**
 * Create a spatial index from points
 */
export function createSpatialIndex(points: ModelPoint[]): SpatialIndex {
  return new SpatialIndex(points);
}

