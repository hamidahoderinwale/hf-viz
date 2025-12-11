/**
 * 3D Force-directed graph visualization showing model relationships.
 * Displays different types of derivatives (finetunes, adapters, quantizations, merges)
 * with color-coded edges and interactive nodes in 3D space.
 */
import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { GraphNode, GraphLink, EdgeType } from './ForceDirectedGraph';
import './ForceDirectedGraph.css';

export interface ForceDirectedGraph3DProps {
  width: number;
  height: number;
  nodes: GraphNode[];
  links: GraphLink[];
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  selectedNodeId?: string | null;
  enabledEdgeTypes?: Set<EdgeType>;
  showLabels?: boolean;
}

// Color scheme for different edge types
const EDGE_COLORS: Record<EdgeType, string> = {
  finetune: '#3b82f6',      // Blue - fine-tuning
  quantized: '#10b981',      // Green - quantization
  adapter: '#f59e0b',         // Orange - adapters
  merge: '#8b5cf6',          // Purple - merges
  parent: '#6b7280',         // Gray - generic parent
};

const EDGE_STROKE_WIDTH: Record<EdgeType, number> = {
  finetune: 2,
  quantized: 1.5,
  adapter: 1.5,
  merge: 2,
  parent: 1,
};

// Simple force simulation for 3D
class ForceSimulation3D {
  private nodes: GraphNode[];
  private links: GraphLink[];
  private velocities: Map<string, THREE.Vector3>;
  private alpha: number;
  private alphaTarget: number;
  private alphaDecay: number;

  constructor(nodes: GraphNode[], links: GraphLink[]) {
    this.nodes = nodes;
    this.links = links;
    this.velocities = new Map();
    this.alpha = 1.0;
    this.alphaTarget = 0;
    this.alphaDecay = 0.0228;

    // Initialize velocities
    nodes.forEach(node => {
      this.velocities.set(node.id, new THREE.Vector3(
        (Math.random() - 0.5) * 0.1,
        (Math.random() - 0.5) * 0.1,
        (Math.random() - 0.5) * 0.1
      ));
    });

    // Initialize positions if not set
    nodes.forEach(node => {
      if (node.x === undefined || node.y === undefined || node.z === undefined) {
        node.x = (Math.random() - 0.5) * 100;
        node.y = (Math.random() - 0.5) * 100;
        node.z = (Math.random() - 0.5) * 100;
      }
    });
  }

  tick() {
    this.alpha += (this.alphaTarget - this.alpha) * 0.1;
    if (this.alpha < 0.001) {
      this.alpha = 0;
      return;
    }

    // Apply forces
    this.applyLinkForce();
    this.applyChargeForce();
    this.applyCenterForce();
    this.updatePositions();
  }

  private applyLinkForce() {
    const linkStrength = 0.1;
    this.links.forEach(link => {
      const source = typeof link.source === 'string' 
        ? this.nodes.find(n => n.id === link.source)
        : link.source;
      const target = typeof link.target === 'string'
        ? this.nodes.find(n => n.id === link.target)
        : link.target;

      if (!source || !target) return;

      const dx = (target.x || 0) - (source.x || 0);
      const dy = (target.y || 0) - (source.y || 0);
      const dz = (target.z || 0) - (source.z || 0);
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

      const edgeType = link.edge_type;
      let idealDistance = 80;
      switch (edgeType) {
        case 'merge':
          idealDistance = 120;
          break;
        case 'finetune':
          idealDistance = 80;
          break;
        case 'quantized':
          idealDistance = 60;
          break;
        case 'adapter':
          idealDistance = 70;
          break;
        default:
          idealDistance = 100;
      }

      const force = (distance - idealDistance) * linkStrength;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      const fz = (dz / distance) * force;

      const sourceVel = this.velocities.get(source.id)!;
      const targetVel = this.velocities.get(target.id)!;

      sourceVel.x += fx;
      sourceVel.y += fy;
      sourceVel.z += fz;
      targetVel.x -= fx;
      targetVel.y -= fy;
      targetVel.z -= fz;
    });
  }

  private applyChargeForce() {
    const chargeStrength = -300;
    const nodes = this.nodes;
    
    // Optimize for large graphs: use Barnes-Hut approximation or limit interactions
    const maxInteractions = nodes.length > 1000 ? 50 : nodes.length;
    
    for (let i = 0; i < nodes.length; i++) {
      const nodeA = nodes[i];
      const velA = this.velocities.get(nodeA.id)!;

      // For large graphs, only interact with nearby nodes
      const interactions = nodes.length > 1000 
        ? this.getNearbyNodes(nodeA, maxInteractions)
        : nodes.slice(i + 1);

      for (const nodeB of interactions) {
        if (nodeB.id === nodeA.id) continue;
        
        const velB = this.velocities.get(nodeB.id)!;

        const dx = (nodeB.x || 0) - (nodeA.x || 0);
        const dy = (nodeB.y || 0) - (nodeA.y || 0);
        const dz = (nodeB.z || 0) - (nodeA.z || 0);
        const distanceSq = dx * dx + dy * dy + dz * dz || 1;
        const distance = Math.sqrt(distanceSq);

        const force = chargeStrength / distanceSq;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        const fz = (dz / distance) * force;

        velA.x -= fx;
        velA.y -= fy;
        velA.z -= fz;
        velB.x += fx;
        velB.y += fy;
        velB.z += fz;
      }
    }
  }

  private getNearbyNodes(node: GraphNode, maxCount: number): GraphNode[] {
    // Simple spatial hash - get nodes within a certain distance
    const nearby: { node: GraphNode; dist: number }[] = [];
    
    for (const other of this.nodes) {
      if (other.id === node.id) continue;
      
      const dx = (other.x || 0) - (node.x || 0);
      const dy = (other.y || 0) - (node.y || 0);
      const dz = (other.z || 0) - (node.z || 0);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      nearby.push({ node: other, dist });
    }
    
    nearby.sort((a, b) => a.dist - b.dist);
    return nearby.slice(0, maxCount).map(n => n.node);
  }

  private applyCenterForce() {
    const centerStrength = 0.01;
    this.nodes.forEach(node => {
      const vel = this.velocities.get(node.id)!;
      vel.x -= (node.x || 0) * centerStrength;
      vel.y -= (node.y || 0) * centerStrength;
      vel.z -= (node.z || 0) * centerStrength;
    });
  }

  private updatePositions() {
    const damping = 0.6;
    this.nodes.forEach(node => {
      const vel = this.velocities.get(node.id)!;
      
      node.x = (node.x || 0) + vel.x * this.alpha;
      node.y = (node.y || 0) + vel.y * this.alpha;
      node.z = (node.z || 0) + vel.z * this.alpha;

      vel.x *= damping;
      vel.y *= damping;
      vel.z *= damping;
    });
  }

  restart() {
    this.alpha = 1.0;
  }
}

function Graph3DScene({
  nodes,
  links,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  enabledEdgeTypes,
  showLabels,
}: ForceDirectedGraph3DProps) {
  const simulationRef = useRef<ForceSimulation3D | null>(null);
  const edgeRefsRef = useRef<Map<string, THREE.BufferGeometry>>(new Map());
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const { raycaster, camera, gl } = useThree();

  // Filter links based on enabled edge types
  const filteredLinks = useMemo((): GraphLink[] => {
    if (!enabledEdgeTypes || enabledEdgeTypes.size === 0) {
      return links;
    }
    return links.filter(link => {
      const linkTypes = link.edge_types || [link.edge_type];
      return linkTypes.some(type => enabledEdgeTypes.has(type));
    });
  }, [links, enabledEdgeTypes]);

  // Filter nodes to only include those connected by filtered links
  const filteredNodes = useMemo(() => {
    if (!enabledEdgeTypes || enabledEdgeTypes.size === 0) {
      return nodes;
    }
    const connectedNodeIds = new Set<string>();
    filteredLinks.forEach(link => {
      const sourceId = typeof link.source === 'string' 
        ? link.source 
        : (link.source as GraphNode).id;
      const targetId = typeof link.target === 'string' 
        ? link.target 
        : (link.target as GraphNode).id;
      connectedNodeIds.add(sourceId);
      connectedNodeIds.add(targetId);
    });
    return nodes.filter(node => connectedNodeIds.has(node.id));
  }, [nodes, filteredLinks, enabledEdgeTypes]);

  // Initialize simulation
  useEffect(() => {
    if (filteredNodes.length === 0) return;

    simulationRef.current = new ForceSimulation3D(filteredNodes, filteredLinks);
    
    // Run simulation for initial layout
    for (let i = 0; i < 100; i++) {
      simulationRef.current.tick();
    }
  }, [filteredNodes, filteredLinks]);

  // Animate simulation - update every frame
  useFrame(() => {
    if (simulationRef.current && simulationRef.current.alpha > 0.001) {
      simulationRef.current.tick();
      
      // Update edge positions
      filteredLinks.forEach((link, idx) => {
        const source = typeof link.source === 'string'
          ? filteredNodes.find(n => n.id === link.source)
          : link.source;
        const target = typeof link.target === 'string'
          ? filteredNodes.find(n => n.id === link.target)
          : link.target;

        if (!source || !target) return;

        const edgeKey = `edge-${idx}`;
        const geometry = edgeRefsRef.current.get(edgeKey);
        if (geometry) {
          const positions = geometry.attributes.position;
          if (positions) {
            positions.array[0] = source.x || 0;
            positions.array[1] = source.y || 0;
            positions.array[2] = source.z || 0;
            positions.array[3] = target.x || 0;
            positions.array[4] = target.y || 0;
            positions.array[5] = target.z || 0;
            positions.needsUpdate = true;
          }
        }
      });
    }
  });

  // Handle node click - handled directly on mesh
  const handleNodeClick = useCallback((node: GraphNode) => {
    if (onNodeClick) {
      onNodeClick(node);
    }
  }, [onNodeClick]);

  // Calculate bounds for camera
  const bounds = useMemo(() => {
    if (filteredNodes.length === 0) {
      return { center: [0, 0, 0], radius: 100 };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    filteredNodes.forEach(node => {
      const x = node.x || 0;
      const y = node.y || 0;
      const z = node.z || 0;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
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
    ) / 2 || 100;

    return { center, radius };
  }, [filteredNodes]);

  if (filteredNodes.length === 0) {
    return null;
  }

  return (
    <>
      {/* Edges - update dynamically */}
      <group>
        {filteredLinks.map((link, idx) => {
          const source = typeof link.source === 'string'
            ? filteredNodes.find(n => n.id === link.source)
            : link.source;
          const target = typeof link.target === 'string'
            ? filteredNodes.find(n => n.id === link.target)
            : link.target;

          if (!source || !target) return null;

          const edgeType = link.edge_type;
          const color = EDGE_COLORS[edgeType] || EDGE_COLORS.parent;
          const width = EDGE_STROKE_WIDTH[edgeType] || 1;

          // Initial positions
          const positions = new Float32Array([
            source.x || 0, source.y || 0, source.z || 0,
            target.x || 0, target.y || 0, target.z || 0,
          ]);

          const edgeKey = `edge-${idx}`;

          return (
            <line key={edgeKey}>
              <bufferGeometry
                ref={(geom) => {
                  if (geom) {
                    edgeRefsRef.current.set(edgeKey, geom);
                  }
                }}
              >
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={positions}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={color} opacity={0.4} transparent linewidth={width} />
            </line>
          );
        })}
      </group>

      {/* Nodes */}
      <group>
        {filteredNodes.map((node) => {
          const downloads = node.downloads || 0;
          const radius = 0.3 + Math.sqrt(downloads) / 8000;
          const isSelected = selectedNodeId === node.id;
          const isHovered = hoveredNodeId === node.id;

          // Color by library
          const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4'];
          const libraries = Array.from(new Set(filteredNodes.map(n => n.library).filter(Boolean)));
          const libIndex = libraries.indexOf(node.library);
          const color = colors[libIndex % colors.length] || '#6b7280';

          return (
            <mesh
              key={node.id}
              position={[node.x || 0, node.y || 0, node.z || 0]}
              userData={{ node }}
              onClick={() => handleNodeClick(node)}
              onPointerEnter={() => {
                setHoveredNodeId(node.id);
                if (onNodeHover) onNodeHover(node);
              }}
              onPointerLeave={() => {
                setHoveredNodeId(null);
                if (onNodeHover) onNodeHover(null);
              }}
            >
              <sphereGeometry args={[radius, 12, 12]} />
              <meshStandardMaterial
                color={isSelected ? '#ef4444' : isHovered ? '#fbbf24' : color}
                emissive={isSelected ? '#ef4444' : isHovered ? '#fbbf24' : color}
                emissiveIntensity={isSelected ? 0.5 : isHovered ? 0.3 : 0.1}
              />
            </mesh>
          );
        })}
      </group>

      {/* Labels - simplified for performance */}
      {showLabels && (
        <group>
          {filteredNodes
            .filter(node => {
              const downloads = node.downloads || 0;
              return downloads > 50000 || selectedNodeId === node.id || hoveredNodeId === node.id;
            })
            .map((node) => (
              <mesh key={`label-${node.id}`} position={[node.x || 0, (node.y || 0) + 3, node.z || 0]}>
                <planeGeometry args={[8, 1.5]} />
                <meshBasicMaterial color="#000" opacity={0.6} transparent />
              </mesh>
            ))}
        </group>
      )}
    </>
  );
}

export default function ForceDirectedGraph3D({
  width,
  height,
  nodes,
  links,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  enabledEdgeTypes,
  showLabels = true,
}: ForceDirectedGraph3DProps) {
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([0, 0, 0]);
  const [cameraTarget, setCameraTarget] = useState<[number, number, number]>([0, 0, 0]);

  // Calculate bounds for camera
  const bounds = useMemo(() => {
    if (nodes.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], radius: 100 };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    nodes.forEach(node => {
      const x = node.x || 0;
      const y = node.y || 0;
      const z = node.z || 0;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
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
    ) / 2 || 100;

    return { center, radius };
  }, [nodes]);

  const handleCameraUpdate = useCallback((position: [number, number, number], target: [number, number, number]) => {
    setCameraPosition(position);
    setCameraTarget(target);
  }, []);

  if (nodes.length === 0) {
    return (
      <div className="force-directed-graph-container">
        <div className="graph-empty">No nodes to display</div>
      </div>
    );
  }

  return (
    <div className="force-directed-graph-container">
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        }}
        camera={{
          position: [
            bounds.center[0] + bounds.radius * 0.5,
            bounds.center[1] + bounds.radius * 0.5,
            bounds.center[2] + bounds.radius * 0.5,
          ],
          fov: 45,
          near: 0.1,
          far: bounds.radius * 20,
        }}
      >
        <color attach="background" args={['#1a1a1a']} />
        
        <OrbitControls
          target={bounds.center}
          enableDamping={true}
          dampingFactor={0.05}
          minDistance={bounds.radius * 0.2}
          maxDistance={bounds.radius * 4}
          makeDefault
        />

        <ambientLight intensity={1.0} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />

        <Graph3DScene
          nodes={nodes}
          links={links}
          onNodeClick={onNodeClick}
          onNodeHover={onNodeHover}
          selectedNodeId={selectedNodeId}
          enabledEdgeTypes={enabledEdgeTypes}
          showLabels={showLabels}
          width={width}
          height={height}
        />
      </Canvas>
    </div>
  );
}
