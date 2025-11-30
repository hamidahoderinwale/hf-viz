import React, { useMemo } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ModelPoint } from '../../types';
import { getCategoricalColorMap, getContinuousColorScale, getDepthColorScale } from '../../utils/rendering/colors';

interface MiniMap3DProps {
  width?: number;
  height?: number;
  data: ModelPoint[];
  colorBy: string;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  onNavigate?: (position: [number, number, number], target: [number, number, number]) => void;
}

// Mini-map point cloud component
function MiniMapPoints({ 
  data, 
  colorBy 
}: { 
  data: ModelPoint[]; 
  colorBy: string;
}) {
  const { positions, colors } = useMemo(() => {
    // Sample for performance (max 1000 points for mini-map)
    const MAX_POINTS = 1000;
    const step = Math.ceil(data.length / MAX_POINTS);
    const sampled: ModelPoint[] = [];
    
    for (let i = 0; i < data.length && sampled.length < MAX_POINTS; i += step) {
      sampled.push(data[i]);
    }
    
    const count = sampled.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    // Calculate color scale
    let colorScale: any;
    
    if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
      const categories = Array.from(new Set(sampled.map((d) => 
        colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown')
      )));
      const colorSchemeType = colorBy === 'library_name' ? 'library' : 'pipeline';
      colorScale = getCategoricalColorMap(categories, colorSchemeType);
    } else if (colorBy === 'downloads' || colorBy === 'likes') {
      const values = sampled.map((d) => colorBy === 'downloads' ? d.downloads : d.likes);
      if (values.length > 0) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        colorScale = getContinuousColorScale(min, max, 'viridis', true);
      }
    } else if (colorBy === 'family_depth') {
      const depths = sampled.map((d) => d.family_depth ?? 0);
      if (depths.length > 0) {
        const maxDepth = Math.max(...depths, 1);
        colorScale = getDepthColorScale(maxDepth, true);
      }
    }
    
    sampled.forEach((model, idx) => {
      positions[idx * 3] = model.x;
      positions[idx * 3 + 1] = model.y;
      positions[idx * 3 + 2] = model.z;
      
      let colorHex = '#60a5fa';
      
      if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
        const value = colorBy === 'library_name' 
          ? (model.library_name || 'unknown') 
          : (model.pipeline_tag || 'unknown');
        colorHex = colorScale instanceof Map ? colorScale.get(value) || '#60a5fa' : '#60a5fa';
      } else if (colorBy === 'downloads' || colorBy === 'likes') {
        const val = colorBy === 'downloads' ? model.downloads : model.likes;
        colorHex = typeof colorScale === 'function' ? colorScale(val) : '#60a5fa';
      } else if (colorBy === 'family_depth') {
        if (typeof colorScale === 'function') {
          colorHex = colorScale(model.family_depth ?? 0);
        }
      }
      
      const color = new THREE.Color(colorHex);
      colors[idx * 3] = color.r;
      colors[idx * 3 + 1] = color.g;
      colors[idx * 3 + 2] = color.b;
    });
    
    return { positions, colors };
  }, [data, colorBy]);
  
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.2}
        vertexColors
        transparent
        opacity={0.7}
        sizeAttenuation={false}
      />
    </points>
  );
}

// Camera indicator showing viewing direction
function CameraIndicator({ 
  position, 
  target 
}: { 
  position: [number, number, number]; 
  target: [number, number, number];
}) {
  // Calculate look direction
  const lookDir = useMemo(() => {
    const dir = new THREE.Vector3(
      target[0] - position[0],
      target[1] - position[1],
      target[2] - position[2]
    ).normalize();
    return dir;
  }, [position, target]);
  
  // Create frustum vertices
  const frustumGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    
    const apex = new THREE.Vector3(...position);
    const forward = lookDir.clone().multiplyScalar(3);
    const right = new THREE.Vector3().crossVectors(lookDir, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, lookDir).normalize();
    
    const baseCenter = apex.clone().add(forward);
    const halfWidth = 1.5;
    const halfHeight = 1;
    
    const corners = [
      baseCenter.clone().add(right.clone().multiplyScalar(halfWidth)).add(up.clone().multiplyScalar(halfHeight)),
      baseCenter.clone().add(right.clone().multiplyScalar(-halfWidth)).add(up.clone().multiplyScalar(halfHeight)),
      baseCenter.clone().add(right.clone().multiplyScalar(-halfWidth)).add(up.clone().multiplyScalar(-halfHeight)),
      baseCenter.clone().add(right.clone().multiplyScalar(halfWidth)).add(up.clone().multiplyScalar(-halfHeight)),
    ];
    
    const vertices = new Float32Array([
      apex.x, apex.y, apex.z, corners[0].x, corners[0].y, corners[0].z,
      apex.x, apex.y, apex.z, corners[1].x, corners[1].y, corners[1].z,
      apex.x, apex.y, apex.z, corners[2].x, corners[2].y, corners[2].z,
      apex.x, apex.y, apex.z, corners[3].x, corners[3].y, corners[3].z,
      corners[0].x, corners[0].y, corners[0].z, corners[1].x, corners[1].y, corners[1].z,
      corners[1].x, corners[1].y, corners[1].z, corners[2].x, corners[2].y, corners[2].z,
      corners[2].x, corners[2].y, corners[2].z, corners[3].x, corners[3].y, corners[3].z,
      corners[3].x, corners[3].y, corners[3].z, corners[0].x, corners[0].y, corners[0].z,
    ]);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    return geometry;
  }, [position, lookDir]);
  
  return (
    <group>
      {/* Camera position sphere */}
      <mesh position={position}>
        <sphereGeometry args={[0.4, 16, 16]} />
        <meshBasicMaterial color="#f97316" />
      </mesh>
      
      {/* Target indicator */}
      <mesh position={target}>
        <sphereGeometry args={[0.2, 8, 8]} />
        <meshBasicMaterial color="#22d3ee" />
      </mesh>
      
      {/* Line from camera to target */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([...position, ...target])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#f97316" opacity={0.7} transparent />
      </line>
      
      {/* Viewing frustum */}
      <lineSegments geometry={frustumGeometry}>
        <lineBasicMaterial color="#f97316" opacity={0.5} transparent />
      </lineSegments>
    </group>
  );
}

// Axis helper for orientation
function AxisHelper() {
  return (
    <group>
      {/* X axis - red */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-20, 0, 0, 20, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ef4444" opacity={0.3} transparent />
      </line>
      
      {/* Y axis - green */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -20, 0, 0, 20, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#22c55e" opacity={0.3} transparent />
      </line>
      
      {/* Z axis - blue */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, -20, 0, 0, 20])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#3b82f6" opacity={0.3} transparent />
      </line>
    </group>
  );
}

// Mini-map camera synced with main view camera
function SyncedMiniMapCamera({ 
  mainCameraPosition, 
  mainCameraTarget 
}: { 
  mainCameraPosition: [number, number, number];
  mainCameraTarget: [number, number, number];
}) {
  const { camera } = useThree();
  
  useFrame(() => {
    // Calculate direction from main camera
    const dir = new THREE.Vector3(
      mainCameraPosition[0] - mainCameraTarget[0],
      mainCameraPosition[1] - mainCameraTarget[1],
      mainCameraPosition[2] - mainCameraTarget[2]
    );
    
    // Scale the distance for the mini-map (fixed overview distance)
    const distance = 30;
    dir.normalize().multiplyScalar(distance);
    
    // Position mini-map camera in same direction but at fixed distance
    camera.position.set(
      mainCameraTarget[0] + dir.x,
      mainCameraTarget[1] + dir.y,
      mainCameraTarget[2] + dir.z
    );
    camera.lookAt(mainCameraTarget[0], mainCameraTarget[1], mainCameraTarget[2]);
    camera.updateProjectionMatrix();
  });
  
  return null;
}

export default function MiniMap3D({
  width = 180,
  height = 140,
  data,
  colorBy,
  cameraPosition,
  cameraTarget,
}: MiniMap3DProps) {
  
  if (data.length === 0) return null;
  
  // Calculate canvas height (subtract header height only)
  const headerHeight = 24;
  const canvasHeight = height - headerHeight;
  
  return (
    <div className="minimap-container minimap-3d" title="3D overview showing your current camera position (orange) and viewing direction in the model space">
      <div className="minimap-header">
        <span className="minimap-title">VIEWPORT</span>
      </div>
      <Canvas
        style={{ width, height: canvasHeight }}
        camera={{ 
          position: [20, 15, 20], 
          fov: 50,
          near: 0.1,
          far: 1000
        }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#0d1117']} />
        
        {/* Ambient light */}
        <ambientLight intensity={0.6} />
        
        {/* Point cloud */}
        <MiniMapPoints data={data} colorBy={colorBy} />
        
        {/* Camera indicator showing user's current view */}
        <CameraIndicator position={cameraPosition} target={cameraTarget} />
        
        {/* Axis helper for orientation */}
        <AxisHelper />
        
        {/* Mini-map camera synced with main view direction */}
        <SyncedMiniMapCamera 
          mainCameraPosition={cameraPosition} 
          mainCameraTarget={cameraTarget} 
        />
      </Canvas>
    </div>
  );
}
