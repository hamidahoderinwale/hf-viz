import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { ModelPoint } from '../../types';
import { createSpatialIndex } from '../../utils/rendering/spatialIndex';
import { adaptiveSampleByDistance } from '../../utils/rendering/frustumCulling';

// WebGL context loss recovery
const handleWebGLContextLoss = (event: Event) => {
  event.preventDefault();
  // Context will be restored automatically by the browser
};

const handleWebGLContextRestored = () => {
  // Context restored - component will re-render automatically
  console.info('WebGL context restored');
};

interface ScatterPlot3DProps {
  data: ModelPoint[];
  colorBy: string;
  sizeBy: string;
  onPointClick?: (model: ModelPoint) => void;
  hoveredModel?: ModelPoint | null;
  onHover?: (model: ModelPoint | null, position?: { x: number; y: number }) => void;
}

function getModelColor(model: ModelPoint, colorBy: string, colorScale: any): string {
  if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
    const value = colorBy === 'library_name' 
      ? (model.library_name || 'unknown') 
      : (model.pipeline_tag || 'unknown');
    return colorScale.get(value) || '#999999';
  } else {
    const val = colorBy === 'downloads' ? model.downloads : model.likes;
    const logVal = Math.log10(val + 1);
    return colorScale(logVal);
  }
}

function getPointSize(model: ModelPoint, sizeBy: string): number {
  if (sizeBy === 'none') return 0.02;
  const val = sizeBy === 'downloads' ? model.downloads : model.likes;
  const logVal = Math.log10(val + 1);
  return 0.01 + (logVal / 7) * 0.04;
}

function Points({ 
  data, 
  colorBy, 
  sizeBy, 
  onPointClick, 
  onHover
}: ScatterPlot3DProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const { camera, size } = useThree();
  const [visiblePoints, setVisiblePoints] = useState<ModelPoint[]>(data);
  const frameCount = useRef(0);

  const colorScale = useMemo(() => {
    if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
      const categories = new Set(data.map((d) => 
        colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown')
      ));
      const colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
      ];
      const scale = new Map();
      Array.from(categories).forEach((cat, i) => {
        scale.set(cat, colors[i % colors.length]);
      });
      return scale;
    } else {
      return (logVal: number) => {
        const t = Math.min(Math.max(logVal / 7, 0), 1);
        if (t < 0.5) {
          const tt = t * 2;
          return `rgb(${Math.floor(tt * 255)}, ${Math.floor(tt * 255)}, ${Math.floor((1 - tt) * 255)})`;
        } else {
          const tt = (t - 0.5) * 2;
          return `rgb(${Math.floor(255)}, ${Math.floor((1 - tt) * 255)}, 0)`;
        }
      };
    }
  }, [data, colorBy]);

  const { positions, colors, sizes, models } = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];
    const models: ModelPoint[] = [];

    visiblePoints.forEach((model) => {
      positions.push(model.x, model.y, model.z);
      
      const color = getModelColor(model, colorBy, colorScale);
      const threeColor = new THREE.Color(color);
      colors.push(threeColor.r, threeColor.g, threeColor.b);
      
      const size = getPointSize(model, sizeBy);
      sizes.push(size);
      
      models.push(model);
    });

    return { positions, colors, sizes, models };
  }, [visiblePoints, colorBy, sizeBy, colorScale]);

  useEffect(() => {
    if (!meshRef.current || positions.length === 0) return;

    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();
    const count = Math.floor(positions.length / 3);

    for (let i = 0; i < count; i++) {
      tempObject.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      tempObject.scale.setScalar(sizes[i]);
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(i, tempObject.matrix);
      
      tempColor.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
      meshRef.current.setColorAt(i, tempColor);
    }

    meshRef.current.count = count;
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [positions, colors, sizes]);

  useEffect(() => {
    if (!meshRef.current || positions.length === 0) return;

    const tempObject = new THREE.Object3D();
    const count = Math.floor(positions.length / 3);
    
    for (let i = 0; i < count; i++) {
      const scale = i === hovered ? sizes[i] * 2 : sizes[i];
      tempObject.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      tempObject.scale.setScalar(scale);
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(i, tempObject.matrix);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [hovered, positions, sizes]);

  useFrame(() => {
    frameCount.current++;
    if (frameCount.current % 10 !== 0) return;
    
    try {
      const sampled = adaptiveSampleByDistance(
        data,
        camera as THREE.Camera,
        1.0,
        50
      );
      
      // Reduced from 100000 to prevent WebGL context loss
      const MAX_RENDER_POINTS = 50000;
      if (sampled.length > MAX_RENDER_POINTS) {
        const step = Math.ceil(sampled.length / MAX_RENDER_POINTS);
        const finalSampled: ModelPoint[] = [];
        for (let i = 0; i < sampled.length; i += step) {
          finalSampled.push(sampled[i]);
        }
        setVisiblePoints(finalSampled);
      } else {
        setVisiblePoints(sampled);
      }
    } catch (error) {
      // Silently handle WebGL errors to prevent console spam
      if (error instanceof Error && error.message.includes('WebGL')) {
        return;
      }
      throw error;
    }
  });

  const handlePointerMove = useCallback((event: any) => {
    event.stopPropagation();
    const instanceId = event.instanceId;
    
    if (instanceId !== undefined && instanceId !== hovered) {
      setHovered(instanceId);
      
      if (onHover && instanceId < models.length) {
        const model = models[instanceId];
        const vector = new THREE.Vector3(
          positions[instanceId * 3],
          positions[instanceId * 3 + 1],
          positions[instanceId * 3 + 2]
        );
        vector.project(camera as THREE.Camera);
        
        const x = (vector.x * 0.5 + 0.5) * size.width;
        const y = (-vector.y * 0.5 + 0.5) * size.height;
        
        onHover(model, { x, y });
      }
    }
  }, [hovered, onHover, models, positions, camera, size]);

  const handlePointerOut = useCallback(() => {
    setHovered(null);
    if (onHover) {
      onHover(null);
    }
  }, [onHover]);

  const handleClick = useCallback((event: any) => {
    event.stopPropagation();
    const instanceId = event.instanceId;
    
    if (onPointClick && instanceId !== undefined && instanceId < models.length) {
      onPointClick(models[instanceId]);
    }
  }, [onPointClick, models]);

  if (visiblePoints.length === 0) return null;

  // Reduced max instances to prevent WebGL context loss
  const maxInstances = Math.min(50000, Math.max(visiblePoints.length, 1000));
  
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, maxInstances]}
      frustumCulled={true}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial vertexColors />
    </instancedMesh>
  );
}

export default function ScatterPlot3D(props: ScatterPlot3DProps) {
  const { data } = props;
  const canvasRef = useRef<HTMLDivElement>(null);

  useMemo(() => {
    if (data.length > 0) {
      createSpatialIndex(data);
    }
  }, [data]);

  // Add WebGL context loss handlers
  useEffect(() => {
    const canvas = canvasRef.current?.querySelector('canvas');
    if (canvas) {
      canvas.addEventListener('webglcontextlost', handleWebGLContextLoss);
      canvas.addEventListener('webglcontextrestored', handleWebGLContextRestored);
      
      return () => {
        canvas.removeEventListener('webglcontextlost', handleWebGLContextLoss);
        canvas.removeEventListener('webglcontextrestored', handleWebGLContextRestored);
      };
    }
  }, []);

  const bounds = useMemo(() => {
    if (data.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], radius: 10 };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    data.forEach((d) => {
      minX = Math.min(minX, d.x);
      maxX = Math.max(maxX, d.x);
      minY = Math.min(minY, d.y);
      maxY = Math.max(maxY, d.y);
      minZ = Math.min(minZ, d.z);
      maxZ = Math.max(maxZ, d.z);
    });

    const center: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];

    const radius = Math.max(
      maxX - minX,
      maxY - minY,
      maxZ - minZ
    ) / 2;

    return { center, radius };
  }, [data]);

  return (
    <div ref={canvasRef} style={{ width: '100%', height: '100%', background: 'var(--background-color)' }}>
      <Canvas
        key="scatter-3d-canvas"
        gl={{ 
          antialias: false,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: false,
          failIfMajorPerformanceCaveat: false,
        }}
        performance={{ min: 0.5 }}
        onCreated={({ gl }) => {
          // Suppress WebGL context loss errors
          gl.domElement.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
          });
        }}
      >
        <PerspectiveCamera
          makeDefault
          position={[
            bounds.center[0] + bounds.radius * 1.5,
            bounds.center[1] + bounds.radius * 1.5,
            bounds.center[2] + bounds.radius * 1.5,
          ]}
          fov={60}
          near={0.1}
          far={bounds.radius * 10}
        />
        
        <OrbitControls
          target={bounds.center}
          enableDamping
          dampingFactor={0.05}
          minDistance={bounds.radius * 0.5}
          maxDistance={bounds.radius * 5}
        />

        <ambientLight intensity={0.8} />
        <directionalLight position={[10, 10, 5]} intensity={0.5} />

        <Points {...props} />

        <gridHelper
          args={[bounds.radius * 4, 20]}
          position={[bounds.center[0], bounds.center[1] - bounds.radius, bounds.center[2]]}
        />
      </Canvas>
    </div>
  );
}
