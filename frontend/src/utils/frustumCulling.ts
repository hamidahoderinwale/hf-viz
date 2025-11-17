/**
 * Frustum culling utilities for camera-based point filtering.
 * Only renders points visible in the camera's view frustum.
 */

import * as THREE from 'three';
import { ModelPoint } from '../types';

/**
 * Calculate camera frustum planes from camera and renderer
 */
export function getFrustumPlanes(camera: THREE.Camera, renderer: THREE.WebGLRenderer): THREE.Plane[] {
  const frustum = new THREE.Frustum();
  const matrix = new THREE.Matrix4();
  matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  frustum.setFromProjectionMatrix(matrix);
  return frustum.planes;
}

/**
 * Check if a point is inside the frustum
 */
export function pointInFrustum(point: ModelPoint, planes: THREE.Plane[]): boolean {
  const vec = new THREE.Vector3(point.x, point.y, point.z);
  for (const plane of planes) {
    if (plane.distanceToPoint(vec) < 0) {
      return false;
    }
  }
  return true;
}

/**
 * Calculate distance from camera to point
 */
export function distanceToCamera(point: ModelPoint, camera: THREE.Camera): number {
  const cameraPos = new THREE.Vector3();
  camera.getWorldPosition(cameraPos);
  const pointPos = new THREE.Vector3(point.x, point.y, point.z);
  return cameraPos.distanceTo(pointPos);
}

/**
 * Calculate level-of-detail factor based on distance
 * Returns a value between 0 and 1, where 1 means full detail
 */
export function calculateLOD(distance: number, maxDistance: number = 10): number {
  if (distance > maxDistance) return 0;
  // Linear falloff
  return 1 - (distance / maxDistance);
}

/**
 * Filter points based on frustum and distance
 */
export function filterVisiblePoints(
  points: ModelPoint[],
  camera: THREE.Camera,
  renderer: THREE.WebGLRenderer,
  maxDistance: number = 10,
  minLOD: number = 0.1
): ModelPoint[] {
  try {
    const planes = getFrustumPlanes(camera, renderer);
    const visible: ModelPoint[] = [];

    for (const point of points) {
      // Check if in frustum
      if (!pointInFrustum(point, planes)) continue;

      // Check distance
      const distance = distanceToCamera(point, camera);
      if (distance > maxDistance) continue;

      // Check LOD
      const lod = calculateLOD(distance, maxDistance);
      if (lod < minLOD) continue;

      visible.push(point);
    }

    return visible;
  } catch (e) {
    // Fallback: return all points if frustum calculation fails
    return points;
  }
}

/**
 * Adaptive sampling based on camera distance
 * Points further away are sampled more aggressively
 */
export function adaptiveSampleByDistance(
  points: ModelPoint[],
  camera: THREE.Camera,
  baseSampleRate: number = 1.0,
  maxDistance: number = 10
): ModelPoint[] {
  const cameraPos = new THREE.Vector3();
  camera.getWorldPosition(cameraPos);

  const sampled: ModelPoint[] = [];
  const distanceBuckets: Map<number, ModelPoint[]> = new Map();

  // Group points by distance
  for (const point of points) {
    const pointPos = new THREE.Vector3(point.x, point.y, point.z);
    const distance = cameraPos.distanceTo(pointPos);
    const bucket = Math.floor(distance / 2); // 2-unit buckets

    if (!distanceBuckets.has(bucket)) {
      distanceBuckets.set(bucket, []);
    }
    distanceBuckets.get(bucket)!.push(point);
  }

  // Sample each bucket with different rates
  const bucketEntries = Array.from(distanceBuckets.entries());
  for (let i = 0; i < bucketEntries.length; i++) {
    const [bucket, bucketPoints] = bucketEntries[i];
    const distance = bucket * 2;
    const lod = calculateLOD(distance, maxDistance);
    const sampleRate = baseSampleRate * lod;

    if (sampleRate >= 1.0) {
      sampled.push(...bucketPoints);
    } else {
      const step = Math.ceil(1 / sampleRate);
      for (let j = 0; j < bucketPoints.length; j += step) {
        sampled.push(bucketPoints[j]);
      }
    }
  }

  return sampled;
}

