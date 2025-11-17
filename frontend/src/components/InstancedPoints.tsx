/**
 * Optimized instanced rendering for 3D points using Three.js InstancedMesh.
 * Much more efficient than rendering individual meshes for large datasets.
 */
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { ModelPoint } from '../types';

interface InstancedPointsProps {
  points: ModelPoint[];
  colors: string[];
  sizes: number[];
  selectedModelId?: string | null;
  familyModelIds?: Set<string>;
  onPointClick?: (model: ModelPoint) => void;
  onHover?: (model: ModelPoint | null) => void;
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
  const { camera, raycaster: globalRaycaster } = useThree();

  // Create geometry and material once
  const geometry = useMemo(() => new THREE.SphereGeometry(0.02, 12, 12), []);
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        metalness: 0.8,
        roughness: 0.2,
        vertexColors: true, // Enable vertex colors for per-instance coloring
        transparent: true,
        opacity: 0.88, // Increased base opacity
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

    // Initialize instanceColor if it doesn't exist
    if (!mesh.instanceColor) {
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

  // Handle raycasting for clicks and hovers, smooth transitions, and depth-based opacity
  useFrame((state, delta) => {
    if (!meshRef.current || !camera) return;

    const mesh = meshRef.current;
    const matrix = new THREE.Matrix4();
    const scale = new THREE.Vector3(1, 1, 1);
    const position = new THREE.Vector3();
    
    // Smooth scale transitions and update depth-based opacity
    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      const isSelected = selectedModelId === point.model_id;
      const isFamilyMember = familyModelIds.has(point.model_id);
      
      // Update target scale for hover
      if (hoveredIndex.current === i) {
        targetScales.current[i] = sizes[i] * 1.5;
      } else {
        targetScales.current[i] = (isSelected || isFamilyMember) ? sizes[i] * 1.5 : sizes[i];
      }
      
      // Smooth scale transition
      currentScales.current[i] += (targetScales.current[i] - currentScales.current[i]) * 0.15;
      
      // Calculate distance from camera for depth-based opacity
      position.set(point.x, point.y, point.z);
      const distance = position.distanceTo(camera.position);
      const maxDistance = 10;
      const minDistance = 1;
      const distanceFactor = Math.max(0.3, Math.min(1, 1 - (distance - minDistance) / (maxDistance - minDistance)));
      
      // Update matrix with smooth scale
      scale.setScalar(currentScales.current[i]);
      matrix.makeScale(scale.x, scale.y, scale.z);
      matrix.setPosition(point.x, point.y, point.z);
      mesh.setMatrixAt(i, matrix);
      
      // Update opacity in instanceColor (we'll need to handle this differently)
      // For now, we'll update the material opacity globally based on average distance
    }
    
    mesh.instanceMatrix.needsUpdate = true;

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
            onHover(points[instanceId]);
          }
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

