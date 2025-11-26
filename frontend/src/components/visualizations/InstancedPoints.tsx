/**
 * Optimized instanced rendering for 3D points using Three.js InstancedMesh.
 * Much more efficient than rendering individual meshes for large datasets.
 */
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { ModelPoint } from '../../types';

interface InstancedPointsProps {
  points: ModelPoint[];
  colors: string[];
  sizes: number[];
  selectedModelId?: string | null;
  familyModelIds?: Set<string>;
  onPointClick?: (model: ModelPoint) => void;
  onHover?: (model: ModelPoint | null, pointer?: { x: number; y: number }) => void;
}

export default function InstancedPoints({
  points,
  colors,
  sizes,
  selectedModelId,
  familyModelIds = new Set(),
  onPointClick,
  onHover,
}: InstancedPointsProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const hoveredIndex = useRef<number | null>(null);
  const targetScales = useRef<Float32Array>(new Float32Array(points.length));
  const currentScales = useRef<Float32Array>(new Float32Array(points.length));
  const { camera } = useThree();

  // Create geometry and material once
  const geometry = useMemo(() => new THREE.SphereGeometry(0.02, 12, 12), []);
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xffffff, // White base color - required for vertexColors to work
        metalness: 0.2, // Reduced for better color visibility
        roughness: 0.6, // Increased for better color visibility
        vertexColors: true, // Enable vertex colors for per-instance coloring
        transparent: true,
        opacity: 0.95, // Increased base opacity for better visibility
        emissive: 0x000000,
        emissiveIntensity: 0.1, // Slight emissive for better visibility
      }),
    []
  );

  // Update instance matrices, colors, and scales
  useEffect(() => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const scale = new THREE.Vector3(1, 1, 1);

    // Initialize or resize instanceColor buffer
    if (!mesh.instanceColor || mesh.instanceColor.count !== points.length) {
      const colorArray = new Float32Array(points.length * 3);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(colorArray, 3);
    }

    // Set up instances
    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      const isSelected = selectedModelId === point.model_id;
      const isFamilyMember = familyModelIds.has(point.model_id);

      // Position
      matrix.setPosition(point.x, point.y, point.z);

      // Color
      color.set(colors[i]);
      if (isSelected) {
        color.set('#ffffff');
      } else if (isFamilyMember) {
        color.set('#4a4a4a');
      }
      mesh.setColorAt(i, color);

      // Scale - initialize target and current scales
      const pointSize = sizes[i];
      const finalSize = isSelected || isFamilyMember ? pointSize * 1.5 : pointSize;
      targetScales.current[i] = finalSize;
      currentScales.current[i] = finalSize;
      scale.setScalar(finalSize);
      matrix.makeScale(finalSize, finalSize, finalSize);
      matrix.setPosition(point.x, point.y, point.z);
      mesh.setMatrixAt(i, matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }, [points, colors, sizes, selectedModelId, familyModelIds]);

  // Track if updates are needed to avoid unnecessary work
  const needsUpdateRef = useRef(true);
  const frameSkipRef = useRef(0);
  
  // Mark as needing update when props change
  useEffect(() => {
    needsUpdateRef.current = true;
  }, [points, colors, sizes, selectedModelId, familyModelIds]);

  // Handle raycasting for clicks and hovers, smooth transitions, and depth-based opacity
  useFrame((state, delta) => {
    if (!meshRef.current || !camera) return;

    const mesh = meshRef.current;
    
    // Adaptive frame skipping based on dataset size and movement
    // More aggressive skipping for better performance, especially for large datasets
    const frameSkip = points.length > 100000 ? 6 : points.length > 50000 ? 4 : points.length > 20000 ? 3 : points.length > 10000 ? 2 : 1;
    frameSkipRef.current++;
    if (frameSkipRef.current % frameSkip !== 0 && !needsUpdateRef.current) {
      return;
    }
    
    // Batch updates - only process visible/important points per frame
    // Smaller batches for smoother performance
    const batchSize = points.length > 100000 ? 3000 : points.length > 50000 ? 5000 : points.length > 20000 ? 8000 : points.length;
    const startIndex = (frameSkipRef.current % Math.ceil(points.length / batchSize)) * batchSize;
    const endIndex = Math.min(startIndex + batchSize, points.length);
    
    const matrix = new THREE.Matrix4();
    const scale = new THREE.Vector3(1, 1, 1);
    const position = new THREE.Vector3();
    const color = new THREE.Color();
    
    let hasChanges = false;
    
    // Smooth scale transitions and update depth-based opacity
    // Process in batches for better performance
    for (let i = startIndex; i < endIndex; i++) {
      const point = points[i];
      const isSelected = selectedModelId === point.model_id;
      const isFamilyMember = familyModelIds.has(point.model_id);
      
      // Update target scale for hover
      const newTargetScale = hoveredIndex.current === i 
        ? sizes[i] * 1.5 
        : (isSelected || isFamilyMember) ? sizes[i] * 1.5 : sizes[i];
      
      if (targetScales.current[i] !== newTargetScale) {
        targetScales.current[i] = newTargetScale;
        hasChanges = true;
      }
      
      // Smooth scale transition (only if target changed or still transitioning)
      const oldScale = currentScales.current[i];
      currentScales.current[i] += (targetScales.current[i] - currentScales.current[i]) * 0.15;
      if (Math.abs(currentScales.current[i] - oldScale) > 0.001) {
        hasChanges = true;
      }
      
      // Only update color if selected/family state changed (optimization)
      // For large datasets, skip distance-based color updates to save performance
      // More aggressive skipping for better performance
      const shouldUpdateColor = points.length < 30000 || isSelected || isFamilyMember || hoveredIndex.current === i;
      
      if (shouldUpdateColor) {
        // Update color with brightness adjustment
        color.set(colors[i]);
        if (isSelected) {
          color.set('#ffffff');
        } else if (isFamilyMember) {
          color.set('#4a4a4a');
        } else {
          // Enhance color vibrancy for better visibility
          const hsl = { h: 0, s: 0, l: 0 };
          color.getHSL(hsl);
          // Increase saturation and lightness slightly
          hsl.s = Math.min(1, hsl.s * 1.15);
          hsl.l = Math.min(0.9, hsl.l * 1.1);
          color.setHSL(hsl.h, hsl.s, hsl.l);
        }
        mesh.setColorAt(i, color);
      }
      
      // Update matrix with smooth scale
      scale.setScalar(currentScales.current[i]);
      matrix.makeScale(scale.x, scale.y, scale.z);
      matrix.setPosition(point.x, point.y, point.z);
      mesh.setMatrixAt(i, matrix);
    }
    
    // Only update if we processed all points or have critical changes
    const processedAll = endIndex >= points.length;
    if (hasChanges || needsUpdateRef.current || processedAll) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor && (needsUpdateRef.current || processedAll || points.length < 50000)) {
        mesh.instanceColor.needsUpdate = true;
      }
      if (processedAll) {
        needsUpdateRef.current = false;
      }
    }

    // Handle hover detection
    if (onHover) {
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(state.pointer, camera);
      const intersects = raycaster.intersectObject(mesh);

      if (intersects.length > 0) {
        const instanceId = intersects[0].instanceId;
        if (instanceId !== undefined && instanceId !== hoveredIndex.current) {
          hoveredIndex.current = instanceId;
          if (instanceId < points.length) {
            // Convert normalized pointer to screen coordinates
            const canvas = state.gl.domElement;
            const rect = canvas.getBoundingClientRect();
            const pointerX = (state.pointer.x * 0.5 + 0.5) * rect.width + rect.left;
            const pointerY = (-state.pointer.y * 0.5 + 0.5) * rect.height + rect.top;
            onHover(points[instanceId], { x: pointerX, y: pointerY });
          }
        } else if (instanceId !== undefined && instanceId === hoveredIndex.current) {
          // Update pointer position while hovering
          const canvas = state.gl.domElement;
          const rect = canvas.getBoundingClientRect();
          const pointerX = (state.pointer.x * 0.5 + 0.5) * rect.width + rect.left;
          const pointerY = (-state.pointer.y * 0.5 + 0.5) * rect.height + rect.top;
          onHover(points[instanceId], { x: pointerX, y: pointerY });
        }
      } else {
        if (hoveredIndex.current !== null) {
          hoveredIndex.current = null;
          onHover(null);
        }
      }
    }
  });

  // Handle clicks
  const handleClick = (event: any) => {
    if (!meshRef.current || !onPointClick || !camera) return;

    event.stopPropagation();
    
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    // Get click position from event
    const rect = event.target.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(meshRef.current);

    if (intersects.length > 0) {
      const instanceId = intersects[0].instanceId;
      if (instanceId !== undefined && instanceId < points.length) {
        onPointClick(points[instanceId]);
      }
    }
  };

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, points.length]}
      frustumCulled={true}
      onClick={handleClick}
    />
  );
}

