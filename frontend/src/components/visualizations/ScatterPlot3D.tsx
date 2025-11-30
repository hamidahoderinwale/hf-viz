import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ModelPoint } from '../../types';
import { getCategoricalColorMap, getContinuousColorScale, getDepthColorScale } from '../../utils/rendering/colors';
import MiniMap3D from './MiniMap3D';
import './ScatterPlot3D.css';
import './MiniMap.css';

interface ScatterPlot3DProps {
  data: ModelPoint[];
  colorBy: string;
  sizeBy: string;
  colorScheme?: string;
  onPointClick?: (model: ModelPoint) => void;
  hoveredModel?: ModelPoint | null;
  onHover?: (model: ModelPoint | null, position?: { x: number; y: number }) => void;
}

function ColoredPoints({ 
  data, 
  colorBy, 
  sizeBy, 
  colorScheme = 'viridis',
  onPointClick,
  isDarkMode = true
}: ScatterPlot3DProps & { isDarkMode?: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);
  const modelLookupRef = useRef<ModelPoint[]>([]);
  const { raycaster, camera, gl } = useThree();

  // Sample and prepare data
  const geometryData = useMemo(() => {
    // Increased limit to support full dataset (150k models)
    const MAX_POINTS = 150000;
    const step = data.length > MAX_POINTS ? Math.ceil(data.length / MAX_POINTS) : 1;
    const sampled: ModelPoint[] = [];
    
    for (let i = 0; i < data.length && sampled.length < MAX_POINTS; i += step) {
      sampled.push(data[i]);
    }

    const count = sampled.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    
    // Calculate color scale
    let colorScale: any = () => '#4a90e2';
    
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
        colorScale = getContinuousColorScale(min, max, colorScheme as any, true);
      }
    } else if (colorBy === 'family_depth') {
      const depths = sampled.map((d) => d.family_depth ?? 0);
      if (depths.length > 0) {
        const maxDepth = Math.max(...depths, 1);
        const uniqueDepths = new Set(depths);
        
        if (uniqueDepths.size <= 2 && maxDepth === 0) {
          const categories = Array.from(new Set(sampled.map((d) => d.library_name || 'unknown')));
          colorScale = getCategoricalColorMap(categories, 'library');
        } else {
          colorScale = getDepthColorScale(maxDepth, isDarkMode);
        }
      }
    }
    
    // Fill arrays
    sampled.forEach((model, idx) => {
      positions[idx * 3] = model.x;
      positions[idx * 3 + 1] = model.y;
      positions[idx * 3 + 2] = model.z;
      
      let colorHex: string;
      if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
        const value = colorBy === 'library_name' 
          ? (model.library_name || 'unknown') 
          : (model.pipeline_tag || 'unknown');
        colorHex = colorScale instanceof Map ? colorScale.get(value) || '#4a90e2' : '#4a90e2';
      } else if (colorBy === 'downloads' || colorBy === 'likes') {
        const val = colorBy === 'downloads' ? model.downloads : model.likes;
        colorHex = typeof colorScale === 'function' ? colorScale(val) : '#4a90e2';
      } else if (colorBy === 'family_depth') {
        if (colorScale instanceof Map) {
          const value = model.library_name || 'unknown';
          colorHex = colorScale.get(value) || '#4a90e2';
        } else if (typeof colorScale === 'function') {
          colorHex = colorScale(model.family_depth ?? 0);
        } else {
          colorHex = '#4a90e2';
        }
      } else {
        colorHex = '#4a90e2';
      }
      
      const color = new THREE.Color(colorHex);
      
      // Preserve original vibrant colors - no washing out
      // The colors from our color utility are already optimized for dark mode
      
      colors[idx * 3] = color.r;
      colors[idx * 3 + 1] = color.g;
      colors[idx * 3 + 2] = color.b;
      
      // Calculate size
      const baseSize = sizeBy === 'none' ? 8 : 6;
      if (sizeBy === 'downloads' || sizeBy === 'likes') {
        const val = sizeBy === 'downloads' ? model.downloads : model.likes;
        const logVal = Math.log10(val + 1);
        sizes[idx] = baseSize + (logVal / 7) * 12;
      } else {
        sizes[idx] = baseSize;
      }
    });
    
    modelLookupRef.current = sampled;
    return { positions, colors, sizes, count };
  }, [data, colorBy, sizeBy, colorScheme, isDarkMode]);

  // Create geometry
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(geometryData.positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(geometryData.colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(geometryData.sizes, 1));
    return geo;
  }, [geometryData]);

  // Create material
  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });
  }, []);

  // Handle click
  const handleClick = (event: any) => {
    if (!onPointClick || !pointsRef.current) return;
    
    // Get mouse position
    const rect = gl.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    
    raycaster.setFromCamera(mouse, camera);
    raycaster.params.Points = { threshold: 0.5 };
    
    const intersects = raycaster.intersectObject(pointsRef.current);
    if (intersects.length > 0 && intersects[0].index !== undefined) {
      const idx = intersects[0].index;
      if (idx < modelLookupRef.current.length) {
        onPointClick(modelLookupRef.current[idx]);
      }
    }
  };

  if (geometryData.count === 0) return null;

  return (
    <points 
      ref={pointsRef}
      geometry={geometry}
      material={material}
      onClick={handleClick}
      frustumCulled={false}
    />
  );
}

// Camera tracking component for mini-map
function CameraTracker({ 
  onCameraUpdate 
}: { 
  onCameraUpdate: (position: [number, number, number], target: [number, number, number]) => void 
}) {
  const { camera, controls } = useThree();
  
  useFrame(() => {
    if (camera && controls) {
      const orbitControls = controls as any;
      const position: [number, number, number] = [camera.position.x, camera.position.y, camera.position.z];
      const target: [number, number, number] = orbitControls.target 
        ? [orbitControls.target.x, orbitControls.target.y, orbitControls.target.z]
        : [0, 0, 0];
      onCameraUpdate(position, target);
    }
  });
  
  return null;
}

export default function ScatterPlot3D(props: ScatterPlot3DProps) {
  const { data, colorBy } = props;
  
  const [canvasBg, setCanvasBg] = useState(() => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return isDark ? '#0a0a0a' : '#ffffff';
  });
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  });
  
  // Camera state for mini-map
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([0, 0, 10]);
  const [cameraTarget, setCameraTarget] = useState<[number, number, number]>([0, 0, 0]);
  
  // Throttle camera updates
  const lastUpdateRef = useRef<number>(0);
  const handleCameraUpdate = useCallback((position: [number, number, number], target: [number, number, number]) => {
    const now = Date.now();
    if (now - lastUpdateRef.current > 100) { // Update every 100ms
      setCameraPosition(position);
      setCameraTarget(target);
      lastUpdateRef.current = now;
    }
  }, []);

  useEffect(() => {
    const updateBg = () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      setCanvasBg(isDark ? '#0a0a0a' : '#ffffff');
      setIsDarkMode(isDark);
    };
    
    updateBg();
    const observer = new MutationObserver(updateBg);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });
    
    return () => observer.disconnect();
  }, []);

  // Simple bounds calculation
  const bounds = useMemo(() => {
    if (data.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], radius: 10 };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    const step = Math.max(1, Math.floor(data.length / 1000));
    for (let i = 0; i < data.length; i += step) {
      const d = data[i];
      if (isFinite(d.x) && isFinite(d.y) && isFinite(d.z)) {
        minX = Math.min(minX, d.x);
        maxX = Math.max(maxX, d.x);
        minY = Math.min(minY, d.y);
        maxY = Math.max(maxY, d.y);
        minZ = Math.min(minZ, d.z);
        maxZ = Math.max(maxZ, d.z);
      }
    }

    if (!isFinite(minX)) {
      return { center: [0, 0, 0] as [number, number, number], radius: 10 };
    }

    const center: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2
    ];
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const radius = Math.max(size / 2, 1);

    return { center, radius };
  }, [data]);

  if (data.length === 0) {
    return (
      <div className={`scatter-3d-empty ${isDarkMode ? 'dark' : 'light'}`}>
        No data to display
      </div>
    );
  }

  return (
    <div className="scatter-3d-container">
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: false,
          failIfMajorPerformanceCaveat: false,
        }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener('webglcontextlost', (event) => {
            event.preventDefault();
          });
          gl.domElement.addEventListener('webglcontextrestored', () => {});
        }}
        camera={{
          position: [
            bounds.center[0] + bounds.radius * 0.5,
            bounds.center[1] + bounds.radius * 0.5,
            bounds.center[2] + bounds.radius * 0.5,
          ],
          fov: 45,
          near: 0.1,
          far: bounds.radius * 20
        }}
      >
        <color attach="background" args={[canvasBg]} />
        
        <OrbitControls
          target={bounds.center}
          enableDamping={true}
          dampingFactor={0.05}
          minDistance={bounds.radius * 0.2}
          maxDistance={bounds.radius * 4}
          makeDefault
        />

        <ambientLight intensity={1.0} />

        <ColoredPoints {...props} isDarkMode={isDarkMode} />
        
        <CameraTracker onCameraUpdate={handleCameraUpdate} />
        
      </Canvas>
      
      {/* Mini-map / Overview Map */}
      <MiniMap3D
        data={data}
        colorBy={colorBy}
        cameraPosition={cameraPosition}
        cameraTarget={cameraTarget}
      />
    </div>
  );
}
