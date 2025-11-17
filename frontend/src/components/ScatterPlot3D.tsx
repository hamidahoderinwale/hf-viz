/**
 * 3D scatter plot using Three.js and React Three Fiber.
 * Interactive 3D latent space navigator with family tree visualization.
 */
/// <reference types="@react-three/fiber" />
import React, { useMemo, useRef, useEffect, useState, useCallback, memo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei';
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
  onHover?: (model: ModelPoint | null) => void;
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
}

interface PointProps {
  position: [number, number, number];
  color: string;
  size: number;
  model: ModelPoint;
  isSelected: boolean;
  isFamilyMember: boolean;
  onClick: () => void;
  onHover?: (model: ModelPoint | null) => void;
}

// Memoized Point component to prevent unnecessary re-renders
const Point = memo(function Point({ position, color, size, model, isSelected, isFamilyMember, onClick, onHover }: PointProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (meshRef.current) {
      // Subtle animation for selected/family members
      if (isSelected || isFamilyMember) {
        meshRef.current.rotation.y += 0.01;
      }
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={onClick}
      onPointerOver={() => {
        setHovered(true);
        if (onHover) onHover(model);
      }}
      onPointerOut={() => {
        setHovered(false);
        if (onHover) onHover(null);
      }}
      scale={hovered || isSelected ? size * 1.5 : size}
      frustumCulled={true} // Only render if in view frustum
    >
      <sphereGeometry args={[0.02, 8, 8]} /> {/* Reduced geometry complexity */}
      <meshStandardMaterial
        color={isSelected ? '#ffffff' : isFamilyMember ? '#4a4a4a' : color}
        emissive={isSelected ? '#ffffff' : isFamilyMember ? '#6a6a6a' : '#000000'}
        emissiveIntensity={isSelected ? 0.5 : isFamilyMember ? 0.2 : 0}
        metalness={0.8}
        roughness={0.2}
        opacity={isSelected || isFamilyMember ? 1 : hovered ? 0.95 : 0.75}  // Increased opacity for better visibility
        transparent
      />
    </mesh>
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
  color: string;
}

function FamilyEdge({ start, end, color }: FamilyEdgeProps) {
  const points = useMemo(() => [new THREE.Vector3(...start), new THREE.Vector3(...end)], [start, end]);
  return (
    <Line
      points={points}
      color={color}
      lineWidth={2}  // Increased from 1 to 2 for better visibility
      dashed={false}
    />
  );
}

function SceneContent({
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

  const { xScale, yScale, zScale, colorScale, sizeScale, familyMap } = useMemo(() => {
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
      // Use selected color scheme for continuous values
      const continuousScale = getContinuousColorScale(min, max, colorScheme);
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'downloads' ? d.downloads : d.likes;
        return continuousScale(val);
      };
    }

    // Size scale
    const sizeValues = sampledData.map(d => {
      if (sizeBy === 'downloads') return d.downloads;
      if (sizeBy === 'likes') return d.likes;
      return 1;
    });
    const sizeMin = Math.min(...sizeValues);
    const sizeMax = Math.max(...sizeValues);
    const sizeRange = sizeMax - sizeMin || 1;
    const sizeScale = (d: ModelPoint) => {
      const val = sizeBy === 'downloads' ? d.downloads : sizeBy === 'likes' ? d.likes : 1;
      return 0.5 + ((val - sizeMin) / sizeRange) * 1.5; // 0.5 to 2.0 scale
    };

    // Family map
    const familyMap = new Map<string, ModelPoint>();
    if (familyTree) {
      familyTree.forEach(model => {
        familyMap.set(model.model_id, model);
      });
    }

    return { xScale, yScale, zScale, colorScale, sizeScale, familyMap };
  }, [sampledData, familyTree, colorBy, sizeBy, colorScheme]);

  // Build family edges
  const familyEdges = useMemo(() => {
    if (!familyTree || familyTree.length === 0) return [];
    
    const edges: Array<{ start: [number, number, number]; end: [number, number, number]; model: ModelPoint }> = [];
    const modelMap = new Map(familyTree.map(m => [m.model_id, m]));

    familyTree.forEach(model => {
      if (model.parent_model && modelMap.has(model.parent_model)) {
        const parent = modelMap.get(model.parent_model)!;
        edges.push({
          start: [xScale(parent.x), yScale(parent.y), zScale(parent.z)],
          end: [xScale(model.x), yScale(model.y), zScale(model.z)],
          model,
        });
      }
    });

    return edges;
  }, [familyTree, xScale, yScale, zScale]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Grid for orientation */}
      <Grid
        args={[10, 10]}
        cellColor="#4a4a4a"
        sectionColor="#6a6a6a"
        cellThickness={0.5}
        sectionThickness={1}
        fadeDistance={5}
        fadeStrength={0.5}
      />

      {/* Family tree edges */}
      {familyEdges.map((edge, i) => (
        <FamilyEdge
          key={i}
          start={edge.start}
          end={edge.end}
          color="#8a8a8a"
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
}

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

