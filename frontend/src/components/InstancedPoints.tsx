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
  const { camera, raycaster: globalRaycaster } = useThree();

  // Create geometry and material once
  const geometry = useMemo(() => new THREE.SphereGeometry(0.02, 8, 8), []);
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        metalness: 0.8,
        roughness: 0.2,
        vertexColors: true, // Enable vertex colors for per-instance coloring
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
      mesh.instanceColor = new THREE.BufferAttribute(colorArray, 3);
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

      // Scale
      const pointSize = sizes[i];
      const finalSize = isSelected || isFamilyMember ? pointSize * 1.5 : pointSize;
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

  // Handle raycasting for clicks and hovers
  useFrame((state) => {
    if (!meshRef.current || !onHover) return;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(state.pointer, camera);

    // Intersect with instanced mesh
    const intersects = raycaster.intersectObject(meshRef.current);

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

