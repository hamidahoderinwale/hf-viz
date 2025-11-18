/**
 * 3D scatter plot using Three.js and React Three Fiber.
 * Interactive 3D latent space navigator with family tree visualization.
 */
/// <reference types="@react-three/fiber" />
import React, { useMemo, useRef, useEffect, useState, useCallback, memo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Line } from '@react-three/drei';
import * as THREE from 'three';
import { ModelPoint } from '../types';
import { getCategoricalColorMap, getContinuousColorScale, getModelColor } from '../utils/colors';
import { createSpatialIndex, SpatialIndex } from '../utils/spatialIndex';
import { filterVisiblePoints, adaptiveSampleByDistance, distanceToCamera } from '../utils/frustumCulling';
import { applySpatialSparsity, getAdaptiveSparsityFactor, calculateAverageDistance } from '../utils/spatialSparsity';
import InstancedPoints from './InstancedPoints';
import ColorLegend from './ColorLegend';

interface ScatterPlot3DProps {
  width: number;
  height: number;
  data: ModelPoint[];
  familyTree?: ModelPoint[];
  colorBy: string;
  sizeBy: string;
  colorScheme?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm';
  showLegend?: boolean;
  showNetworkEdges?: boolean;
  showStructuralGroups?: boolean;
  overviewMode?: boolean;
  networkEdgeType?: 'library' | 'pipeline' | 'combined';
  onPointClick?: (model: ModelPoint) => void;
  selectedModelId?: string | null;
  onViewChange?: (center: { x: number; y: number; z: number }) => void;
  onHover?: (model: ModelPoint | null, pointer?: { x: number; y: number }) => void;
  targetViewCenter?: { x: number; y: number; z: number } | null; // Target position to animate to
}

interface PointProps {
  position: [number, number, number];
  color: string;
  size: number;
  model: ModelPoint;
  isSelected: boolean;
  isFamilyMember: boolean;
  onClick: () => void;
  onHover?: (model: ModelPoint | null, pointer?: { x: number; y: number }) => void;
}

// Memoized Point component with enhanced visual effects
const Point = memo(function Point({ position, color, size, model, isSelected, isFamilyMember, onClick, onHover }: PointProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const outlineRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const { camera } = useThree();
  
  // Smooth size transition
  const targetScale = useRef(size);
  const currentScale = useRef(size);
  
  // Update target scale when hover/selection changes
  useEffect(() => {
    targetScale.current = (hovered || isSelected) ? size * 1.5 : size;
  }, [hovered, isSelected, size]);

  useFrame(() => {
    if (!meshRef.current || !camera) return;
    
    // Smooth size transition using lerp
    currentScale.current += (targetScale.current - currentScale.current) * 0.15;
    meshRef.current.scale.setScalar(currentScale.current);
    
    // Calculate distance from camera for depth-based opacity
    const distance = meshRef.current.position.distanceTo(camera.position);
    const maxDistance = 10;
    const minDistance = 1;
    const distanceFactor = Math.max(0.3, Math.min(1, 1 - (distance - minDistance) / (maxDistance - minDistance)));
    
    // Update opacity based on distance
    if (meshRef.current.material instanceof THREE.MeshStandardMaterial) {
      const baseOpacity = isSelected || isFamilyMember ? 1 : hovered ? 0.95 : 0.88;
      meshRef.current.material.opacity = baseOpacity * distanceFactor;
    }
    
      // Subtle animation for selected/family members
      if (isSelected || isFamilyMember) {
        meshRef.current.rotation.y += 0.01;
    }
    
    // Glow effect for selected/hovered points
    if (glowRef.current) {
      glowRef.current.scale.setScalar(currentScale.current * 1.5);
      if (glowRef.current.material instanceof THREE.MeshStandardMaterial) {
        glowRef.current.material.opacity = (isSelected ? 0.4 : hovered ? 0.2 : 0) * distanceFactor;
      }
    }
    
    // Outline effect
    if (outlineRef.current) {
      outlineRef.current.scale.setScalar(currentScale.current * 1.1);
      if (outlineRef.current.material instanceof THREE.MeshBasicMaterial) {
        outlineRef.current.material.opacity = (isSelected ? 0.8 : hovered ? 0.5 : 0) * distanceFactor;
      }
    }
  });

  return (
    <group>
      {/* Glow effect */}
      {(isSelected || hovered) && (
        <mesh ref={glowRef} position={position}>
          <sphereGeometry args={[0.02, 16, 16]} />
          <meshStandardMaterial
            color={isSelected ? '#ffffff' : color}
            emissive={isSelected ? '#ffffff' : color}
            emissiveIntensity={1}
            transparent
            opacity={0}
            side={THREE.BackSide}
          />
        </mesh>
      )}
      
      {/* Outline effect */}
      {(isSelected || hovered) && (
        <mesh ref={outlineRef} position={position}>
          <sphereGeometry args={[0.02, 16, 16]} />
          <meshBasicMaterial
            color={isSelected ? '#ffffff' : '#ffffff'}
            transparent
            opacity={0}
            side={THREE.BackSide}
            depthWrite={false}
          />
        </mesh>
      )}
      
      {/* Main point */}
    <mesh
      ref={meshRef}
      position={position}
      onClick={onClick}
        onPointerOver={(e) => {
        setHovered(true);
          if (onHover) {
            onHover(model, { 
              x: e.clientX, 
              y: e.clientY 
            });
          }
      }}
      onPointerOut={() => {
        setHovered(false);
        if (onHover) onHover(null);
      }}
        onPointerMove={(e) => {
          if (hovered && onHover) {
            onHover(model, { 
              x: e.clientX, 
              y: e.clientY 
            });
          }
        }}
        frustumCulled={true}
      >
        <sphereGeometry args={[0.02, 12, 12]} />
      <meshStandardMaterial
        color={isSelected ? '#ffffff' : isFamilyMember ? '#4a4a4a' : color}
        emissive={isSelected ? '#ffffff' : isFamilyMember ? '#6a6a6a' : '#000000'}
          emissiveIntensity={isSelected ? 0.6 : isFamilyMember ? 0.2 : 0}
          metalness={0.3}
          roughness={0.7}
          opacity={0.9}
        transparent
      />
    </mesh>
    </group>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function for memo
  return (
    prevProps.model.model_id === nextProps.model.model_id &&
    prevProps.isSelected === nextProps.isSelected &&
    prevProps.isFamilyMember === nextProps.isFamilyMember &&
    prevProps.color === nextProps.color &&
    prevProps.size === nextProps.size &&
    prevProps.position[0] === nextProps.position[0] &&
    prevProps.position[1] === nextProps.position[1] &&
    prevProps.position[2] === nextProps.position[2]
  );
});

interface FamilyEdgeProps {
  start: [number, number, number];
  end: [number, number, number];
  parentColor: string;
  childColor: string;
  depth: number;
}

function FamilyEdge({ start, end, parentColor, childColor, depth }: FamilyEdgeProps) {
  const points = useMemo(() => [new THREE.Vector3(...start), new THREE.Vector3(...end)], [start, end]);
  const flowRef = useRef<THREE.Mesh>(null);
  const flowProgress = useRef(0);
  
  // Interpolate color between parent and child for gradient effect
  const interpolatedColor = useMemo(() => {
    const parent = new THREE.Color(parentColor);
    const child = new THREE.Color(childColor);
    const mid = new THREE.Color().lerpColors(parent, child, 0.5);
    return `#${mid.getHexString()}`;
  }, [parentColor, childColor]);
  
  // Depth-based opacity and thickness
  const depthFactor = Math.min(depth / 5, 1);
  const opacity = 0.6 + depthFactor * 0.4;
  const lineWidth = 2.5 + depth * 0.4;
  
  // Animated flow along edge
  useFrame((state, delta) => {
    if (!flowRef.current) return;
    
    flowProgress.current += delta * 0.5; // Flow speed
    if (flowProgress.current > 1) flowProgress.current = 0;
    
    // Position flow particle along the edge
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3().subVectors(endVec, startVec);
    const position = new THREE.Vector3().addVectors(startVec, direction.multiplyScalar(flowProgress.current));
    
    flowRef.current.position.copy(position);
    
    // Pulse effect
    const pulse = Math.sin(flowProgress.current * Math.PI * 2) * 0.3 + 0.7;
    flowRef.current.scale.setScalar(pulse);
  });
  
  return (
    <group>
      {/* Main edge with interpolated color - thicker and more visible */}
    <Line
      points={points}
        color={interpolatedColor}
        lineWidth={lineWidth}
      dashed={false}
        transparent
        opacity={opacity}
      />
      
      {/* Animated flow particle */}
      <mesh ref={flowRef} position={start}>
        <sphereGeometry args={[0.015, 8, 8]} />
        <meshBasicMaterial
          color={childColor}
          transparent
          opacity={0.8}
        />
      </mesh>
    </group>
  );
}

// Memoize SceneContent to prevent unnecessary re-renders
const SceneContent = memo(function SceneContent({
  data,
  familyTree,
  colorBy,
  sizeBy,
  colorScheme = 'viridis',
  showNetworkEdges = false,
  showStructuralGroups = false,
  overviewMode = false,
  networkEdgeType = 'combined',
  onPointClick,
  selectedModelId,
  onHover,
  isInteracting = false,
}: Omit<ScatterPlot3DProps, 'width' | 'height' | 'showLegend'> & { isInteracting?: boolean }) {
  const { camera, gl } = useThree();
  const [useInstancedRendering, setUseInstancedRendering] = useState(false);
  const cameraPositionRef = useRef<THREE.Vector3>(new THREE.Vector3());
  const lastCameraPositionRef = useRef<THREE.Vector3>(new THREE.Vector3());
  const movementSpeedRef = useRef<number>(0);

  // Track camera movement for adaptive quality
  useFrame(() => {
    if (camera) {
      camera.getWorldPosition(cameraPositionRef.current);
      const movement = cameraPositionRef.current.distanceTo(lastCameraPositionRef.current);
      movementSpeedRef.current = movement;
      lastCameraPositionRef.current.copy(cameraPositionRef.current);
      
      // Use instanced rendering for large datasets (more efficient for >5K points)
      // Always use for datasets > 10K, or when moving fast with >5K points
      setUseInstancedRendering(data.length > 5000 || (data.length > 1000 && movementSpeedRef.current > 0.01));
    }
  });

  // Create spatial index for efficient queries
  const spatialIndex = useMemo(() => {
    if (data.length === 0) return null;
    return createSpatialIndex(data);
  }, [data]);

  // Level-of-detail: sample data for performance when there are too many points
  // Use camera-based culling, adaptive sampling, and spatial sparsity
  const sampledData = useMemo(() => {
    if (data.length === 0) return [];
    
    // Keep all family tree members and selected models
    const familyIds = new Set(familyTree?.map(m => m.model_id) || []);
    const importantIds = new Set<string>();
    const important: ModelPoint[] = [];
    const others: ModelPoint[] = [];
    
    for (const d of data) {
      if (familyIds.has(d.model_id) || selectedModelId === d.model_id) {
        important.push(d);
        importantIds.add(d.model_id);
      } else {
        others.push(d);
      }
    }

    // For very large datasets, use spatial indexing and camera-based culling
    // Use instanced rendering for datasets > 10K points
    if (data.length > 10000 && spatialIndex && camera && gl) {
      // Use adaptive sampling based on distance from camera
      // When moving fast, reduce quality for better performance
      const qualityFactor = isInteracting && movementSpeedRef.current > 0.01 ? 0.7 : 1.0; // Increased from 0.6
      const maxDistance = 20 * qualityFactor; // Increased view distance from 15 to 20
      
      // Improved sampling strategy to show more models while maintaining performance
      // Use higher sample rates to better represent the full dataset
      let distanceSampled: ModelPoint[];
      if (others.length > 400000) {
        // For extremely large datasets (>400K), sample 30% (increased from 15%)
        const sampleRate = qualityFactor * 0.3;
        const step = Math.ceil(1 / sampleRate);
        distanceSampled = [];
        for (let i = 0; i < others.length; i += step) {
          distanceSampled.push(others[i]);
        }
      } else if (others.length > 200000) {
        // For very large datasets (200K-400K), sample 40% (increased from 15%)
        const sampleRate = qualityFactor * 0.4;
        const step = Math.ceil(1 / sampleRate);
        distanceSampled = [];
        for (let i = 0; i < others.length; i += step) {
          distanceSampled.push(others[i]);
        }
      } else if (others.length > 100000) {
        // For large datasets (100K-200K), sample 50% (increased from 20%)
        const sampleRate = qualityFactor * 0.5;
        const step = Math.ceil(1 / sampleRate);
        distanceSampled = [];
        for (let i = 0; i < others.length; i += step) {
          distanceSampled.push(others[i]);
        }
      } else {
        // Use adaptive sampling with higher rate for better representation
        distanceSampled = adaptiveSampleByDistance(others, camera, qualityFactor * 0.85, maxDistance); // Increased from 0.7
      }
      
      // Apply frustum culling if camera is available
      // Increased limit for instanced rendering (can handle more with better sampling)
      const maxVisible = Math.min(distanceSampled.length, 200000); // Increased from 100K to 200K
      let visible: ModelPoint[];
      try {
        visible = filterVisiblePoints(
          distanceSampled.slice(0, maxVisible), 
          camera, 
          gl, 
          maxDistance, 
          0.01 // Lower LOD threshold to show more points (was 0.02)
        );
      } catch (e) {
        // Fallback if frustum calculation fails
        visible = distanceSampled.slice(0, maxVisible);
      }
      
      // Apply spatial sparsity to reduce density and improve navigability
      // But be less aggressive to show more models
      const combined = [...important, ...visible];
      if (combined.length > 5000) { // Increased threshold from 3000
        // Calculate adaptive minimum distance based on data spread
        const avgDistance = calculateAverageDistance(combined);
        const sparsityFactor = getAdaptiveSparsityFactor(combined.length) * 1.2; // Reduced from 1.5 to show more
        const minDistance = avgDistance * sparsityFactor;
        
        if (minDistance > 0) {
          return applySpatialSparsity(combined, minDistance, importantIds);
        }
      }
      
      return combined;
    }
    
    // For smaller datasets, use simple sampling with sparsity
    // Increased render limit to show more models
    const renderLimit = data.length > 200000 ? 200000 : data.length; // Increased from 100K to 200K
    if (data.length <= renderLimit) {
      // Still apply sparsity even if under limit for better navigability
      // But be less aggressive to show more models
      if (data.length > 5000) { // Increased threshold from 3000
        const avgDistance = calculateAverageDistance(data);
        const sparsityFactor = getAdaptiveSparsityFactor(data.length) * 1.2; // Reduced from 1.5
        const minDistance = avgDistance * sparsityFactor;
        if (minDistance > 0) {
          return applySpatialSparsity(data, minDistance, importantIds);
        }
      }
      return data;
    }
    
    // Sample from others, keep all important
    const sampleSize = Math.min(renderLimit - important.length, others.length);
    let sampled: ModelPoint[];
    if (others.length > sampleSize) {
      // Use efficient sampling - use every Nth element for better distribution
      const step = Math.ceil(others.length / sampleSize);
      sampled = [];
      for (let i = 0; i < others.length && sampled.length < sampleSize; i += step) {
        sampled.push(others[i]);
      }
    } else {
      sampled = others;
    }
    
    // Apply spatial sparsity to reduce density
    const combined = [...important, ...sampled];
    if (combined.length > 3000) { // Lower threshold
      const avgDistance = calculateAverageDistance(combined);
      const sparsityFactor = getAdaptiveSparsityFactor(combined.length) * 1.5; // Increase sparsity
      const minDistance = avgDistance * sparsityFactor;
      
      if (minDistance > 0) {
        return applySpatialSparsity(combined, minDistance, importantIds);
      }
    }
    
    return combined;
  }, [data, familyTree, selectedModelId, spatialIndex, camera, gl, isInteracting]);

  // Cache scales to avoid recalculation
  const scalesCacheRef = useRef<{
    dataLength: number;
    colorBy: string;
    sizeBy: string;
    colorScheme: string;
    scales: any;
  } | null>(null);

  const { xScale, yScale, zScale, colorScale, sizeScale, familyMap } = useMemo(() => {
    // Return cached scales if inputs haven't changed
    if (scalesCacheRef.current &&
        scalesCacheRef.current.dataLength === sampledData.length &&
        scalesCacheRef.current.colorBy === colorBy &&
        scalesCacheRef.current.sizeBy === sizeBy &&
        scalesCacheRef.current.colorScheme === colorScheme) {
      return scalesCacheRef.current.scales;
    }
    if (sampledData.length === 0) {
      return {
        xScale: (x: number) => x,
        yScale: (y: number) => y,
        zScale: (z: number) => z,
        colorScale: () => '#808080',
        sizeScale: () => 1,
        familyMap: new Map<string, ModelPoint>(),
      };
    }

    const xExtent = [Math.min(...sampledData.map(d => d.x)), Math.max(...sampledData.map(d => d.x))] as [number, number];
    const yExtent = [Math.min(...sampledData.map(d => d.y)), Math.max(...sampledData.map(d => d.y))] as [number, number];
    const zExtent = [Math.min(...sampledData.map(d => d.z)), Math.max(...sampledData.map(d => d.z))] as [number, number];

    // Normalize to [-1, 1] range for better 3D visualization
    const xRange = xExtent[1] - xExtent[0] || 1;
    const yRange = yExtent[1] - yExtent[0] || 1;
    const zRange = zExtent[1] - zExtent[0] || 1;

    const xScale = (x: number) => ((x - xExtent[0]) / xRange - 0.5) * 2;
    const yScale = (y: number) => ((y - yExtent[0]) / yRange - 0.5) * 2;
    const zScale = (z: number) => ((z - zExtent[0]) / zRange - 0.5) * 2;

    // Color scale with improved color schemes
    const isCategorical = colorBy === 'library_name' || colorBy === 'pipeline_tag' || colorBy === 'cluster_id';
    let colorScale: (d: ModelPoint) => string;

    if (colorBy === 'cluster_id') {
      // Color by cluster - use distinct colors for each cluster
      const clusters = Array.from(new Set(sampledData.map(d => d.cluster_id).filter(id => id !== null))) as number[];
      const colorMap = getCategoricalColorMap(clusters.map(String), 'default');
      colorScale = (d: ModelPoint) => {
        return d.cluster_id !== null ? colorMap.get(String(d.cluster_id)) || '#808080' : '#808080';
      };
    } else if (colorBy === 'family_depth') {
      // Color by family depth - use sequential color scale (darker = deeper)
      const depths = sampledData.map(d => d.family_depth ?? 0);
      const maxDepth = Math.max(...depths, 1);
      const continuousScale = getContinuousColorScale(0, maxDepth, colorScheme);
      colorScale = (d: ModelPoint) => {
        const depth = d.family_depth ?? 0;
        return continuousScale(depth);
      };
    } else if (colorBy === 'licenses') {
      // Color by license type (categorical)
      const licenses = Array.from(new Set(sampledData.map(d => {
        if (!d.licenses) return 'No License';
        const licenseStr = d.licenses.toString();
        // Extract first license from string
        try {
          if (licenseStr.startsWith('[')) {
            const cleaned = licenseStr.replace(/'/g, '"');
            const parsed = JSON.parse(cleaned);
            return Array.isArray(parsed) && parsed.length > 0 ? parsed[0] : 'No License';
          }
          return licenseStr.split(',')[0].trim() || 'No License';
        } catch {
          return licenseStr.split(',')[0].trim() || 'No License';
        }
      })));
      const colorMap = getCategoricalColorMap(licenses, 'default');
      colorScale = (d: ModelPoint) => {
        if (!d.licenses) return colorMap.get('No License') || '#808080';
        const licenseStr = d.licenses.toString();
        try {
          if (licenseStr.startsWith('[')) {
            const cleaned = licenseStr.replace(/'/g, '"');
            const parsed = JSON.parse(cleaned);
            const firstLicense = Array.isArray(parsed) && parsed.length > 0 ? parsed[0] : 'No License';
            return colorMap.get(firstLicense) || '#808080';
          }
          const firstLicense = licenseStr.split(',')[0].trim() || 'No License';
          return colorMap.get(firstLicense) || '#808080';
        } catch {
          const firstLicense = licenseStr.split(',')[0].trim() || 'No License';
          return colorMap.get(firstLicense) || '#808080';
        }
      };
    } else if (colorBy === 'trending_score') {
      // Color by trending score
      const scores = sampledData.map(d => d.trending_score ?? 0).filter(s => s !== null);
      if (scores.length > 0) {
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const continuousScale = getContinuousColorScale(min, max, colorScheme);
        colorScale = (d: ModelPoint) => {
          const score = d.trending_score ?? 0;
          return continuousScale(score);
        };
      } else {
        colorScale = () => '#808080';
      }
    } else if (isCategorical) {
      const categories = Array.from(new Set(sampledData.map(d => {
        if (colorBy === 'library_name') return d.library_name || 'unknown';
        return d.pipeline_tag || 'unknown';
      })));
      const colorScheme = colorBy === 'library_name' ? 'library' : 'pipeline';
      const colorMap = getCategoricalColorMap(categories, colorScheme);
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'library_name' ? d.library_name : d.pipeline_tag;
        return colorMap.get(val || 'unknown') || '#808080';
      };
    } else {
      const values = sampledData.map(d => colorBy === 'downloads' ? d.downloads : d.likes);
      const min = Math.min(...values);
      const max = Math.max(...values);
      // Use logarithmic scaling for downloads/likes (heavily skewed distributions)
      // This provides better visual representation of the data
      const useLogScale = colorBy === 'downloads' || colorBy === 'likes';
      const continuousScale = getContinuousColorScale(min, max, colorScheme, useLogScale);
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'downloads' ? d.downloads : d.likes;
        return continuousScale(val);
      };
    }

    // Size scale with logarithmic scaling for better representation of skewed distributions
    const sizeValues = sampledData.map(d => {
      if (sizeBy === 'downloads') return d.downloads;
      if (sizeBy === 'likes') return d.likes;
      return 1;
    });
    const sizeMin = Math.min(...sizeValues);
    const sizeMax = Math.max(...sizeValues);
    // Use logarithmic scaling for downloads/likes to better represent the distribution
    const useLogSize = sizeBy === 'downloads' || sizeBy === 'likes';
    const logSizeMin = useLogSize && sizeMin > 0 ? Math.log10(sizeMin + 1) : sizeMin;
    const logSizeMax = useLogSize && sizeMax > 0 ? Math.log10(sizeMax + 1) : sizeMax;
    const logSizeRange = logSizeMax - logSizeMin || 1;
    const sizeRange = sizeMax - sizeMin || 1;
    const sizeScale = (d: ModelPoint) => {
      let normalizedSize: number;
      const val = sizeBy === 'downloads' ? d.downloads : sizeBy === 'likes' ? d.likes : 1;
      if (useLogSize && val > 0) {
        const logVal = Math.log10(val + 1);
        normalizedSize = (logVal - logSizeMin) / logSizeRange;
      } else {
        normalizedSize = (val - sizeMin) / sizeRange;
      }
      // Scale from 0.5 to 3.0 for better visibility
      return 0.5 + Math.max(0, Math.min(1, normalizedSize)) * 2.5;
    };

    // Family map
    const familyMap = new Map<string, ModelPoint>();
    if (familyTree) {
      familyTree.forEach(model => {
        familyMap.set(model.model_id, model);
      });
    }

    const scales = { xScale, yScale, zScale, colorScale, sizeScale, familyMap };
    
    // Cache the scales
    scalesCacheRef.current = {
      dataLength: sampledData.length,
      colorBy,
      sizeBy,
      colorScheme,
      scales,
    };
    
    return scales;
  }, [sampledData, familyTree, colorBy, sizeBy, colorScheme]);

  // Build family edges with color coding by depth
  const familyEdges = useMemo(() => {
    if (!familyTree || familyTree.length === 0) return [];
    
    const edges: Array<{ 
      start: [number, number, number]; 
      end: [number, number, number]; 
      model: ModelPoint;
      parentColor: string;
      childColor: string;
      depth: number;
    }> = [];
    const modelMap = new Map(familyTree.map(m => [m.model_id, m]));
    
    // Get color scale for family depth
    const maxDepth = Math.max(...familyTree.map(m => m.family_depth ?? 0), 1);
    const depthColorScale = getContinuousColorScale(0, maxDepth, colorScheme);

    familyTree.forEach(model => {
      if (model.parent_model && modelMap.has(model.parent_model)) {
        const parent = modelMap.get(model.parent_model)!;
        const parentDepth = parent.family_depth ?? 0;
        const childDepth = model.family_depth ?? 0;
        
        // Color based on depth - parent to child gradient
        const parentColor = depthColorScale(parentDepth);
        const childColor = depthColorScale(childDepth);
        
        edges.push({
          start: [xScale(parent.x), yScale(parent.y), zScale(parent.z)],
          end: [xScale(model.x), yScale(model.y), zScale(model.z)],
          model,
          parentColor,
          childColor,
          depth: childDepth,
        });
      }
    });

    return edges;
  }, [familyTree, xScale, yScale, zScale, colorScheme]);

  // Build network edges (co-occurrence relationships)
  const networkEdges = useMemo(() => {
    if (!showNetworkEdges || sampledData.length === 0) return [];
    
    const edges: Array<{
      start: [number, number, number];
      end: [number, number, number];
      weight: number;
      type: string;
    }> = [];
    
    // Create a map for quick lookups
    const modelMap = new Map(sampledData.map(m => [m.model_id, m]));
    
    // Group models by library, pipeline, or both
    const groups = new Map<string, ModelPoint[]>();
    
    sampledData.forEach(model => {
      if (networkEdgeType === 'library' || networkEdgeType === 'combined') {
        const key = `lib:${model.library_name || 'unknown'}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key)!.push(model);
      }
      if (networkEdgeType === 'pipeline' || networkEdgeType === 'combined') {
        const key = `pipe:${model.pipeline_tag || 'unknown'}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key)!.push(model);
      }
    });
    
    // Create edges between models in the same group
    // Limit to avoid performance issues - only connect nearby models
    const maxConnectionsPerModel = overviewMode ? 5 : 3;
    const maxDistance = overviewMode ? 0.5 : 0.3; // Distance threshold in normalized space
    
    groups.forEach((models, groupKey) => {
      if (models.length < 2) return;
      
      // For large groups, sample connections
      const modelsToConnect = models.length > 50 
        ? models.filter((_, i) => i % Math.ceil(models.length / 50) === 0)
        : models;
      
      for (let i = 0; i < modelsToConnect.length; i++) {
        const model1 = modelsToConnect[i];
        let connections = 0;
        
        for (let j = i + 1; j < modelsToConnect.length && connections < maxConnectionsPerModel; j++) {
          const model2 = modelsToConnect[j];
          
          // Calculate distance in normalized space
          const dx = model1.x - model2.x;
          const dy = model1.y - model2.y;
          const dz = model1.z - model2.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (distance < maxDistance) {
            edges.push({
              start: [xScale(model1.x), yScale(model1.y), zScale(model1.z)],
              end: [xScale(model2.x), yScale(model2.y), zScale(model2.z)],
              weight: 1 - (distance / maxDistance), // Higher weight for closer models
              type: groupKey.startsWith('lib:') ? 'library' : 'pipeline',
            });
            connections++;
          }
        }
      }
    });
    
    return edges;
  }, [showNetworkEdges, sampledData, networkEdgeType, overviewMode, xScale, yScale, zScale]);

  // Build structural groupings (library/pipeline clusters)
  const structuralGroups = useMemo(() => {
    if (!showStructuralGroups || sampledData.length === 0) return [];
    
    const groups: Array<{
      models: ModelPoint[];
      type: 'library' | 'pipeline';
      name: string;
      color: string;
      center: [number, number, number];
    }> = [];
    
    // Group by library
    const libraryGroups = new Map<string, ModelPoint[]>();
    sampledData.forEach(model => {
      const lib = model.library_name || 'unknown';
      if (!libraryGroups.has(lib)) libraryGroups.set(lib, []);
      libraryGroups.get(lib)!.push(model);
    });
    
    // Group by pipeline
    const pipelineGroups = new Map<string, ModelPoint[]>();
    sampledData.forEach(model => {
      const pipe = model.pipeline_tag || 'unknown';
      if (!pipelineGroups.has(pipe)) pipelineGroups.set(pipe, []);
      pipelineGroups.get(pipe)!.push(model);
    });
    
    // Only show groups with multiple models and reasonable size
    const minGroupSize = overviewMode ? 3 : 5;
    const maxGroups = overviewMode ? 20 : 10;
    
    // Process library groups
    const sortedLibGroups = Array.from(libraryGroups.entries())
      .filter(([_, models]) => models.length >= minGroupSize)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, maxGroups);
    
    sortedLibGroups.forEach(([name, models]) => {
      // Calculate center
      const centerX = models.reduce((sum, m) => sum + m.x, 0) / models.length;
      const centerY = models.reduce((sum, m) => sum + m.y, 0) / models.length;
      const centerZ = models.reduce((sum, m) => sum + m.z, 0) / models.length;
      
      groups.push({
        models,
        type: 'library',
        name,
        color: '#4a90e2',
        center: [xScale(centerX), yScale(centerY), zScale(centerZ)],
      });
    });
    
    // Process pipeline groups
    const sortedPipeGroups = Array.from(pipelineGroups.entries())
      .filter(([_, models]) => models.length >= minGroupSize)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, maxGroups);
    
    sortedPipeGroups.forEach(([name, models]) => {
      const centerX = models.reduce((sum, m) => sum + m.x, 0) / models.length;
      const centerY = models.reduce((sum, m) => sum + m.y, 0) / models.length;
      const centerZ = models.reduce((sum, m) => sum + m.z, 0) / models.length;
      
      groups.push({
        models,
        type: 'pipeline',
        name,
        color: '#e24a90',
        center: [xScale(centerX), yScale(centerY), zScale(centerZ)],
      });
    });
    
    return groups;
  }, [showStructuralGroups, sampledData, overviewMode, xScale, yScale, zScale]);

  // Adjust camera for overview mode
  useEffect(() => {
    if (!camera) return;
    
    const currentPos = new THREE.Vector3();
    camera.getWorldPosition(currentPos);
    const distance = currentPos.length();
    
    if (overviewMode) {
      // Move camera further back to see more of the scene
      const newDistance = Math.max(distance, 8); // Minimum distance for overview
      
      if (newDistance > distance) {
        // Smoothly animate camera back
        const targetPos = currentPos.clone().normalize().multiplyScalar(newDistance);
        const startPos = currentPos.clone();
        const duration = 1000; // 1 second
        const startTime = Date.now();
        
        const animate = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
          
          const pos = startPos.clone().lerp(targetPos, eased);
          camera.position.copy(pos);
          
          if (progress < 1) {
            requestAnimationFrame(animate);
          }
        };
        
        animate();
      }
    }
    // Note: When overview mode is disabled, user can manually zoom back in
    // We don't force camera position to avoid disrupting user's navigation
  }, [overviewMode, camera]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Grid for orientation - using custom grid to avoid deprecation warnings */}
      <gridHelper args={[10, 10, '#6a6a6a', '#4a4a4a']} />

      {/* Network edges (co-occurrence relationships) */}
      {networkEdges.length > 0 && (
        <group>
          {networkEdges.slice(0, overviewMode ? networkEdges.length : Math.min(networkEdges.length, 5000)).map((edge, i) => (
            <Line
              key={`network-${i}`}
              points={[new THREE.Vector3(...edge.start), new THREE.Vector3(...edge.end)]}
              color={edge.type === 'library' ? '#4a90e2' : '#e24a90'}
              lineWidth={overviewMode ? 1.5 : 1}
              transparent
              opacity={overviewMode ? 0.2 * edge.weight : 0.3 * edge.weight}
              dashed={false}
            />
          ))}
        </group>
      )}

      {/* Structural groupings - show cluster centers and boundaries */}
      {structuralGroups.map((group, i) => {
        // Calculate bounding sphere radius from center
        let maxRadius = 0;
        group.models.forEach(model => {
          const modelPos = [
            xScale(model.x),
            yScale(model.y),
            zScale(model.z)
          ];
          const dx = modelPos[0] - group.center[0];
          const dy = modelPos[1] - group.center[1];
          const dz = modelPos[2] - group.center[2];
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          maxRadius = Math.max(maxRadius, distance);
        });
        
        // Add some padding
        const radius = Math.max(maxRadius * 1.2, 0.2);

        return (
          <group key={`group-${group.type}-${i}`}>
            {/* Group center marker */}
            <mesh position={group.center}>
              <sphereGeometry args={[0.05, 8, 8]} />
              <meshBasicMaterial color={group.color} transparent opacity={0.6} />
            </mesh>
            {/* Bounding sphere (wireframe) */}
            <mesh position={group.center}>
              <sphereGeometry args={[radius, 16, 16]} />
              <meshBasicMaterial 
                color={group.color} 
                wireframe 
                transparent 
                opacity={0.15}
              />
            </mesh>
          </group>
        );
      })}

      {/* Family tree edges with gradient and animation */}
      {familyEdges.map((edge, i) => (
        <FamilyEdge
          key={`${edge.model.model_id}-${i}`}
          start={edge.start}
          end={edge.end}
          parentColor={edge.parentColor}
          childColor={edge.childColor}
          depth={edge.depth}
        />
      ))}

      {/* Data points - use instanced rendering for large datasets */}
      {useInstancedRendering && sampledData.length > 1000 ? (
        <InstancedPoints
          points={sampledData.map(m => ({
            ...m,
            x: xScale(m.x),
            y: yScale(m.y),
            z: zScale(m.z),
          }))}
          colors={sampledData.map(m => colorScale(m))}
          sizes={sampledData.map(m => sizeScale(m))}
          selectedModelId={selectedModelId}
          familyModelIds={new Set(familyTree?.map(m => m.model_id) || [])}
          onPointClick={onPointClick}
          onHover={onHover}
        />
      ) : (
        sampledData.map((model) => {
        const isFamilyMember = familyMap.has(model.model_id);
        const isSelected = selectedModelId === model.model_id;
        
        return (
          <Point
            key={model.model_id}
            position={[xScale(model.x), yScale(model.y), zScale(model.z)]}
            color={colorScale(model)}
            size={sizeScale(model)}
            model={model}
            isSelected={isSelected}
            isFamilyMember={isFamilyMember}
            onClick={() => onPointClick?.(model)}
            onHover={onHover}
          />
        );
        })
      )}

      {/* Axes helper */}
      <axesHelper args={[2]} />
    </>
  );
}, (prevProps, nextProps) => {
  // Custom comparison to prevent unnecessary re-renders
  return (
    prevProps.data.length === nextProps.data.length &&
    prevProps.colorBy === nextProps.colorBy &&
    prevProps.sizeBy === nextProps.sizeBy &&
    prevProps.colorScheme === nextProps.colorScheme &&
    prevProps.selectedModelId === nextProps.selectedModelId &&
    prevProps.isInteracting === nextProps.isInteracting &&
    prevProps.showNetworkEdges === nextProps.showNetworkEdges &&
    prevProps.showStructuralGroups === nextProps.showStructuralGroups &&
    prevProps.overviewMode === nextProps.overviewMode &&
    prevProps.networkEdgeType === nextProps.networkEdgeType &&
    (prevProps.familyTree?.length || 0) === (nextProps.familyTree?.length || 0)
  );
});

// Component to track interaction state
function InteractionTracker({ 
  controlsRef, 
  onInteractionChange 
}: { 
  controlsRef: React.RefObject<any>;
  onInteractionChange: (isInteracting: boolean) => void;
}) {
  useEffect(() => {
    if (!controlsRef.current) return;
    
    const controls = controlsRef.current;
    let interactionTimeout: NodeJS.Timeout;
    
    const handleStart = () => {
      onInteractionChange(true);
      clearTimeout(interactionTimeout);
    };
    
    const handleEnd = () => {
      interactionTimeout = setTimeout(() => {
        onInteractionChange(false);
      }, 100);
    };
    
    // Track interaction events
    const domElement = controls.domElement;
    domElement.addEventListener('mousedown', handleStart);
    domElement.addEventListener('mousemove', handleStart);
    domElement.addEventListener('wheel', handleStart);
    domElement.addEventListener('mouseup', handleEnd);
    domElement.addEventListener('mouseleave', handleEnd);
    
    return () => {
      domElement.removeEventListener('mousedown', handleStart);
      domElement.removeEventListener('mousemove', handleStart);
      domElement.removeEventListener('wheel', handleStart);
      domElement.removeEventListener('mouseup', handleEnd);
      domElement.removeEventListener('mouseleave', handleEnd);
      clearTimeout(interactionTimeout);
    };
  }, [controlsRef, onInteractionChange]);
  
  return null;
}

export default function ScatterPlot3D({
  width,
  height,
  data,
  familyTree,
  colorBy,
  sizeBy,
  colorScheme = 'viridis',
  showLegend = true,
  showNetworkEdges = false,
  showStructuralGroups = false,
  overviewMode = false,
  networkEdgeType = 'combined',
  onPointClick,
  selectedModelId,
  onViewChange,
  onHover,
  targetViewCenter,
}: ScatterPlot3DProps) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);
  const [isInteracting, setIsInteracting] = useState(false);
  const previousTargetRef = useRef<{ x: number; y: number; z: number } | null>(null);

  // Animate camera to target view center when it changes
  useEffect(() => {
    if (!targetViewCenter || !controlsRef.current || !cameraRef.current) return;
    
    // Check if target has actually changed
    if (previousTargetRef.current &&
        previousTargetRef.current.x === targetViewCenter.x &&
        previousTargetRef.current.y === targetViewCenter.y &&
        previousTargetRef.current.z === targetViewCenter.z) {
      return;
    }
    
    previousTargetRef.current = { ...targetViewCenter };
    
    // Animate camera to target position
    const controls = controlsRef.current;
    const camera = cameraRef.current;
    
    // Calculate camera position relative to target
    // Position camera at a good viewing distance from the target
    const distance = 3; // Distance from target
    const target = new THREE.Vector3(targetViewCenter.x, targetViewCenter.y, targetViewCenter.z);
    const cameraPosition = new THREE.Vector3(
      target.x + distance,
      target.y + distance,
      target.z + distance
    );
    
    // Smoothly animate to target
    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const duration = 1000; // 1 second animation
    const startTime = Date.now();
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (ease-in-out)
      const eased = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      
      // Interpolate camera position
      camera.position.lerpVectors(startPosition, cameraPosition, eased);
      controls.target.lerpVectors(startTarget, target, eased);
      controls.update();
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    animate();
  }, [targetViewCenter]);

  // Update view center when camera changes
  useEffect(() => {
    if (!cameraRef.current || !onViewChange) return;

    const updateViewCenter = () => {
      if (cameraRef.current && controlsRef.current) {
        // Get the orbit controls target (center of view)
        const target = controlsRef.current.target;
        const center = {
          x: target.x,
          y: target.y,
          z: target.z,
        };
        onViewChange(center);
      }
    };

    // Update on camera changes
    const interval = setInterval(updateViewCenter, 100);
    return () => clearInterval(interval);
  }, [onViewChange]);

  return (
    <div style={{ width, height, background: '#ffffff', position: 'relative' }}>
      {showLegend && data.length > 0 && (
        <ColorLegend 
          colorBy={colorBy} 
          data={data} 
          position="top-right"
        />
      )}
      <Canvas
        camera={{ position: [3, 3, 3], fov: 50 }}
        gl={{ antialias: !isInteracting, alpha: true }} // Disable antialiasing when interacting for performance
        performance={{ min: 0.5 }} // Target 50% performance budget
      >
        <PerspectiveCamera makeDefault ref={cameraRef} position={[3, 3, 3]} fov={50} />
        <OrbitControls
          ref={controlsRef}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1}
          maxDistance={overviewMode ? 15 : 10}
          enableDamping={true}
          dampingFactor={0.05}
        />
        <InteractionTracker controlsRef={controlsRef} onInteractionChange={setIsInteracting} />
        <SceneContent
          data={data}
          familyTree={familyTree}
          colorBy={colorBy}
          sizeBy={sizeBy}
          colorScheme={colorScheme}
          showNetworkEdges={showNetworkEdges}
          showStructuralGroups={showStructuralGroups}
          overviewMode={overviewMode}
          networkEdgeType={networkEdgeType}
          onPointClick={onPointClick}
          selectedModelId={selectedModelId}
          onHover={onHover}
          isInteracting={isInteracting}
        />
      </Canvas>
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          fontSize: '11px',
          color: '#6a6a6a',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '4px 8px',
          borderRadius: '2px',
          border: '1px solid #d0d0d0',
          fontFamily: "'Vend Sans', sans-serif",
        }}
      >
        <strong>3D Navigation:</strong> Click + drag to rotate | Scroll to zoom | Right-click + drag to pan
      </div>
      {(overviewMode || showNetworkEdges || showStructuralGroups) && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            right: 10,
            fontSize: '11px',
            color: '#2c5f2d',
            backgroundColor: 'rgba(240, 248, 240, 0.95)',
            padding: '6px 10px',
            borderRadius: '4px',
            border: '1px solid #90ee90',
            fontFamily: "'Vend Sans', sans-serif",
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          {overviewMode && (
            <div style={{ fontWeight: '600' }}>🔍 Overview Mode Active</div>
          )}
          {showNetworkEdges && (
            <div style={{ fontSize: '10px' }}>
              Network: {networkEdgeType === 'combined' ? 'Library + Pipeline' : networkEdgeType}
            </div>
          )}
          {showStructuralGroups && (
            <div style={{ fontSize: '10px' }}>Structural Groups: Library & Pipeline clusters</div>
          )}
        </div>
      )}
    </div>
  );
}

