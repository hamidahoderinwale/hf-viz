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
      const qualityFactor = isInteracting && movementSpeedRef.current > 0.01 ? 0.6 : 1.0;
      const maxDistance = 15 * qualityFactor; // Increased view distance
      
      // Sample based on distance - more aggressive for very large datasets
      let distanceSampled: ModelPoint[];
      if (others.length > 200000) {
        // For extremely large datasets (>200K), use more aggressive sampling for sparsity
        const sampleRate = qualityFactor * 0.15; // Sample 15% when not interacting (was 30%)
        const step = Math.ceil(1 / sampleRate);
        distanceSampled = [];
        for (let i = 0; i < others.length; i += step) {
          distanceSampled.push(others[i]);
        }
      } else if (others.length > 100000) {
        // For large datasets (100K-200K), sample 20%
        const sampleRate = qualityFactor * 0.2;
        const step = Math.ceil(1 / sampleRate);
        distanceSampled = [];
        for (let i = 0; i < others.length; i += step) {
          distanceSampled.push(others[i]);
        }
      } else {
        // Use adaptive sampling with reduced rate for more sparsity
        distanceSampled = adaptiveSampleByDistance(others, camera, qualityFactor * 0.7, maxDistance);
      }
      
      // Apply frustum culling if camera is available
      // Increased limit for instanced rendering (can handle more)
      const maxVisible = Math.min(distanceSampled.length, 100000); // Increased from 20K to 100K
      let visible: ModelPoint[];
      try {
        visible = filterVisiblePoints(
          distanceSampled.slice(0, maxVisible), 
          camera, 
          gl, 
          maxDistance, 
          0.02 // Lower LOD threshold to show more points
        );
      } catch (e) {
        // Fallback if frustum calculation fails
        visible = distanceSampled.slice(0, maxVisible);
      }
      
      // Apply spatial sparsity to reduce density and improve navigability
      const combined = [...important, ...visible];
      if (combined.length > 3000) { // Lower threshold for sparsity application
        // Calculate adaptive minimum distance based on data spread
        const avgDistance = calculateAverageDistance(combined);
        const sparsityFactor = getAdaptiveSparsityFactor(combined.length) * 1.5; // Increase sparsity by 50%
        const minDistance = avgDistance * sparsityFactor;
        
        if (minDistance > 0) {
          return applySpatialSparsity(combined, minDistance, importantIds);
        }
      }
      
      return combined;
    }
    
    // For smaller datasets, use simple sampling with sparsity
    // Reduced render limit for better sparsity and navigability
    const renderLimit = data.length > 100000 ? 100000 : data.length; // Reduced from 200K to 100K
    if (data.length <= renderLimit) {
      // Still apply sparsity even if under limit for better navigability
      if (data.length > 3000) {
        const avgDistance = calculateAverageDistance(data);
        const sparsityFactor = getAdaptiveSparsityFactor(data.length) * 1.5;
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

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Grid for orientation - using custom grid to avoid deprecation warnings */}
      <gridHelper args={[10, 10, '#6a6a6a', '#4a4a4a']} />

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
          maxDistance={10}
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
    </div>
  );
}

