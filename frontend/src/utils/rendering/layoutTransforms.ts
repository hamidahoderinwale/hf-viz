/**
 * Layout transformation utilities for different rendering styles.
 * Transforms 3D coordinates based on selected rendering style.
 */
import { ModelPoint } from '../../types';

export type RenderingStyle = 'embeddings' | 'sphere' | 'galaxy' | 'wave' | 'helix' | 'torus';

/**
 * Transform model coordinates based on rendering style.
 */
export function transformLayout(
  model: ModelPoint,
  style: RenderingStyle,
  index: number,
  total: number
): [number, number, number] {
  const { x, y, z } = model;
  
  switch (style) {
    case 'embeddings':
      // Use original embedding coordinates
      return [x, y, z];
    
    case 'sphere':
      // Map to sphere surface
      const radius = 3;
      const theta = Math.atan2(y, x); // Azimuthal angle
      const phi = Math.acos(z / Math.sqrt(x * x + y * y + z * z)); // Polar angle
      return [
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi)
      ];
    
    case 'galaxy':
      // Spiral galaxy pattern
      const galaxyRadius = 2 + (index / total) * 2;
      const galaxyAngle = (index / total) * Math.PI * 8; // Multiple spirals
      const galaxyZ = (index / total - 0.5) * 2;
      return [
        galaxyRadius * Math.cos(galaxyAngle),
        galaxyRadius * Math.sin(galaxyAngle),
        galaxyZ
      ];
    
    case 'wave':
      // Wave pattern based on original coordinates
      const waveX = x;
      const waveY = y;
      const waveZ = Math.sin(x * 2) * Math.cos(y * 2) * 1.5;
      return [waveX, waveY, waveZ];
    
    case 'helix':
      // Helical arrangement
      const helixRadius = 2;
      const helixAngle = (index / total) * Math.PI * 4;
      const helixZ = (index / total - 0.5) * 4;
      return [
        helixRadius * Math.cos(helixAngle),
        helixRadius * Math.sin(helixAngle),
        helixZ
      ];
    
    case 'torus':
      // Torus/donut shape
      const majorRadius = 2.5;
      const minorRadius = 1;
      const torusAngle = (index / total) * Math.PI * 2;
      const torusTubeAngle = (index / total) * Math.PI * 4;
      return [
        (majorRadius + minorRadius * Math.cos(torusTubeAngle)) * Math.cos(torusAngle),
        (majorRadius + minorRadius * Math.cos(torusTubeAngle)) * Math.sin(torusAngle),
        minorRadius * Math.sin(torusTubeAngle)
      ];
    
    default:
      return [x, y, z];
  }
}

/**
 * Get geometry type for rendering style.
 */
export function getGeometryType(style: RenderingStyle): 'sphere' | 'torus' | 'box' {
  switch (style) {
    case 'torus':
      return 'torus';
    case 'sphere':
    case 'galaxy':
    case 'helix':
      return 'sphere';
    default:
      return 'sphere';
  }
}

