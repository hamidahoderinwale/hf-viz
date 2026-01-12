/**
 * Highly optimized 3D Force-directed graph visualization using instanced rendering.
 * Designed to handle millions of nodes efficiently by:
 * 1. Using pre-computed positions from server
 * 2. Instanced mesh rendering (single draw call for all nodes)
 * 3. GPU-based picking for interactions
 * 4. Level-of-detail (LOD) for labels
 */
import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { GraphNode, GraphLink, EdgeType } from './ForceDirectedGraph';
import { getContinuousColorScale, LIBRARY_COLORS, PIPELINE_COLORS } from '../../utils/rendering/colors';
import './ForceDirectedGraph.css';

export type ColorByOption = 'library' | 'pipeline' | 'downloads' | 'likes' | 'edge_type';
export type SizeByOption = 'downloads' | 'likes' | 'uniform';
export type ColorScheme = 'viridis' | 'plasma' | 'inferno' | 'coolwarm';

export interface ForceDirectedGraph3DInstancedProps {
  width: number;
  height: number;
  nodes: GraphNode[];
  links: GraphLink[];
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  selectedNodeId?: string | null;
  enabledEdgeTypes?: Set<EdgeType>;
  showLabels?: boolean;
  maxVisibleNodes?: number;
  maxVisibleEdges?: number;
  linkDistance?: number;
  chargeStrength?: number;
  collisionRadius?: number;
  nodeSizeMultiplier?: number;
  edgeOpacity?: number;
  colorBy?: ColorByOption;
  sizeBy?: SizeByOption;
  colorScheme?: ColorScheme;
  highlightedNodeId?: string | null;
  familyFilter?: string;
  searchQuery?: string;
}

// Color scheme for different edge types
const EDGE_COLORS: Record<EdgeType, THREE.Color> = {
  finetune: new THREE.Color('#3b82f6'),      // Blue
  quantized: new THREE.Color('#10b981'),      // Green
  adapter: new THREE.Color('#f59e0b'),         // Orange
  merge: new THREE.Color('#8b5cf6'),          // Purple
  parent: new THREE.Color('#6b7280'),         // Gray
};

/**
 * Get color for a node based on colorBy option
 */
function getNodeColorByOption(
  node: GraphNode,
  colorBy: ColorByOption,
  colorScale?: (value: number) => string
): THREE.Color {
  if (colorBy === 'downloads' && colorScale) {
    return new THREE.Color(colorScale(node.downloads || 0));
  }
  if (colorBy === 'likes' && colorScale) {
    return new THREE.Color(colorScale(node.likes || 0));
  }
  if (colorBy === 'pipeline') {
    const colorHex = PIPELINE_COLORS[node.pipeline?.toLowerCase() || 'unknown'] || '#6b7280';
    return new THREE.Color(colorHex);
  }
  // Default: library
  const colorHex = LIBRARY_COLORS[node.library?.toLowerCase() || 'unknown'] || '#6b7280';
  return new THREE.Color(colorHex);
}

/**
 * Calculate node size based on sizeBy option
 */
function getNodeSizeByOption(node: GraphNode, sizeBy: SizeByOption): number {
  if (sizeBy === 'downloads') {
    return 0.3 + Math.log10(Math.max(node.downloads || 1, 1)) * 0.15;
  }
  if (sizeBy === 'likes') {
    return 0.3 + Math.log10(Math.max(node.likes || 1, 1)) * 0.2;
  }
  // uniform
  return 0.5;
}

/**
 * Instanced nodes component - renders all nodes in a single draw call
 */
function InstancedNodes({
  nodes,
  selectedNodeId,
  onNodeClick,
  onNodeHover,
  maxVisible = 500000,
  nodeSizeMultiplier = 1.0,
  colorBy = 'library',
  sizeBy = 'downloads',
  colorScheme = 'viridis',
  highlightedNodeId,
  familyFilter,
  searchQuery,
}: {
  nodes: GraphNode[];
  selectedNodeId?: string | null;
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  maxVisible?: number;
  nodeSizeMultiplier?: number;
  colorBy?: ColorByOption;
  sizeBy?: SizeByOption;
  colorScheme?: ColorScheme;
  highlightedNodeId?: string | null;
  familyFilter?: string;
  searchQuery?: string;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera, raycaster, pointer } = useThree();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  
  // Filter nodes by family and search query first
  const preFilteredNodes = useMemo(() => {
    let result = nodes;
    
    // Filter by family (organization prefix)
    if (familyFilter && familyFilter.trim()) {
      const filter = familyFilter.toLowerCase();
      result = result.filter(node => {
        const nodeId = node.id.toLowerCase();
        return nodeId.startsWith(filter + '/') || nodeId.includes('/' + filter + '/');
      });
    }
    
    // Filter by search query
    if (searchQuery && searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(node => 
        node.id.toLowerCase().includes(query) ||
        node.title?.toLowerCase().includes(query)
      );
    }
    
    return result;
  }, [nodes, familyFilter, searchQuery]);
  
  // Limit nodes for performance
  const visibleNodes = useMemo(() => {
    if (preFilteredNodes.length <= maxVisible) return preFilteredNodes;
    // Sort by downloads and take top N
    return [...preFilteredNodes]
      .sort((a, b) => (b.downloads || 0) - (a.downloads || 0))
      .slice(0, maxVisible);
  }, [preFilteredNodes, maxVisible]);
  
  // Node ID to index map for lookup
  const nodeIndexMap = useMemo(() => {
    const map = new Map<string, number>();
    visibleNodes.forEach((node, i) => map.set(node.id, i));
    return map;
  }, [visibleNodes]);
  
  // Create color scale for continuous colorBy options
  const colorScale = useMemo(() => {
    if (colorBy === 'downloads' || colorBy === 'likes') {
      const values = visibleNodes.map(n => colorBy === 'downloads' ? (n.downloads || 0) : (n.likes || 0));
      const min = Math.min(...values, 0);
      const max = Math.max(...values, 1);
      return getContinuousColorScale(min, max, colorScheme, true);
    }
    return undefined;
  }, [colorBy, colorScheme, visibleNodes]);
  
  // Pre-compute matrices and colors
  const { matrices, colors, sizes } = useMemo(() => {
    const matrices: THREE.Matrix4[] = [];
    const colors: THREE.Color[] = [];
    const sizes: number[] = [];
    
    const tempMatrix = new THREE.Matrix4();
    
    visibleNodes.forEach((node) => {
      const x = node.x || 0;
      const y = node.y || 0;
      const z = node.z || 0;
      const size = getNodeSizeByOption(node, sizeBy) * nodeSizeMultiplier;
      
      tempMatrix.makeScale(size, size, size);
      tempMatrix.setPosition(x, y, z);
      matrices.push(tempMatrix.clone());
      
      colors.push(getNodeColorByOption(node, colorBy, colorScale));
      sizes.push(size);
    });
    
    return { matrices, colors, sizes };
  }, [visibleNodes, colorBy, sizeBy, colorScale, nodeSizeMultiplier]);
  
  // Update instance attributes when data changes
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    
    const tempColor = new THREE.Color();
    
    matrices.forEach((matrix, i) => {
      mesh.setMatrixAt(i, matrix);
      
      // Highlight selected/hovered/highlighted nodes
      const isSelected = visibleNodes[i]?.id === selectedNodeId;
      const isHighlighted = visibleNodes[i]?.id === highlightedNodeId;
      const isHovered = i === hoveredIndex;
      
      if (isSelected) {
        tempColor.set('#ef4444'); // Red for selected
      } else if (isHighlighted) {
        tempColor.set('#22d3ee'); // Cyan for highlighted (search result)
      } else if (isHovered) {
        tempColor.set('#fbbf24'); // Yellow for hovered
      } else {
        tempColor.copy(colors[i]);
      }
      
      mesh.setColorAt(i, tempColor);
    });
    
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [matrices, colors, selectedNodeId, highlightedNodeId, hoveredIndex, visibleNodes]);
  
  // Raycasting for hover/click
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(mesh);
    
    if (intersects.length > 0) {
      const index = intersects[0].instanceId;
      if (index !== undefined && index !== hoveredIndex) {
        setHoveredIndex(index);
        if (onNodeHover && visibleNodes[index]) {
          onNodeHover(visibleNodes[index]);
        }
      }
    } else if (hoveredIndex !== null) {
      setHoveredIndex(null);
      if (onNodeHover) {
        onNodeHover(null);
      }
    }
  });
  
  // Handle click
  const handleClick = useCallback(() => {
    if (hoveredIndex !== null && onNodeClick && visibleNodes[hoveredIndex]) {
      onNodeClick(visibleNodes[hoveredIndex]);
    }
  }, [hoveredIndex, onNodeClick, visibleNodes]);
  
  if (visibleNodes.length === 0) return null;
  
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, visibleNodes.length]}
      onClick={handleClick}
      frustumCulled={false}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial
        vertexColors
        roughness={0.4}
        metalness={0.1}
      />
    </instancedMesh>
  );
}

/**
 * Edges component using line segments
 */
function Edges({
  nodes,
  links,
  enabledEdgeTypes,
  maxVisible = 100000,
  edgeOpacity = 0.6,
  familyFilter,
  searchQuery,
}: {
  nodes: GraphNode[];
  links: GraphLink[];
  enabledEdgeTypes?: Set<EdgeType>;
  maxVisible?: number;
  edgeOpacity?: number;
  familyFilter?: string;
  searchQuery?: string;
}) {
  const lineRef = useRef<THREE.LineSegments>(null);
  
  // Filter nodes by family and search query first
  const filteredNodeIds = useMemo(() => {
    let result = nodes;
    
    // Filter by family (organization prefix)
    if (familyFilter && familyFilter.trim()) {
      const filter = familyFilter.toLowerCase();
      result = result.filter(node => {
        const nodeId = node.id.toLowerCase();
        return nodeId.startsWith(filter + '/') || nodeId.includes('/' + filter + '/');
      });
    }
    
    // Filter by search query
    if (searchQuery && searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(node => 
        node.id.toLowerCase().includes(query) ||
        node.title?.toLowerCase().includes(query)
      );
    }
    
    return new Set(result.map(n => n.id));
  }, [nodes, familyFilter, searchQuery]);
  
  // Create node lookup map
  const nodeMap = useMemo(() => {
    const map = new Map<string, GraphNode>();
    nodes.forEach(node => map.set(node.id, node));
    return map;
  }, [nodes]);
  
  // Filter and limit links
  const visibleLinks = useMemo(() => {
    let filtered = links;
    
    // Filter by edge types
    if (enabledEdgeTypes && enabledEdgeTypes.size > 0) {
      filtered = links.filter(link => {
        const linkTypes = link.edge_types || [link.edge_type];
        return linkTypes.some(type => enabledEdgeTypes.has(type));
      });
    }
    
    // Filter to only include links where both source and target are in filtered nodes
    if (familyFilter || searchQuery) {
      filtered = filtered.filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : link.source?.id;
        const targetId = typeof link.target === 'string' ? link.target : link.target?.id;
        return filteredNodeIds.has(sourceId || '') && filteredNodeIds.has(targetId || '');
      });
    }
    
    if (filtered.length > maxVisible) {
      return filtered.slice(0, maxVisible);
    }
    
    return filtered;
  }, [links, enabledEdgeTypes, maxVisible, familyFilter, searchQuery, filteredNodeIds]);
  
  // Build geometry
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    
    visibleLinks.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source?.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target?.id;
      
      const source = nodeMap.get(sourceId || '');
      const target = nodeMap.get(targetId || '');
      
      if (!source || !target) return;
      
      // Source position
      positions.push(source.x || 0, source.y || 0, source.z || 0);
      // Target position
      positions.push(target.x || 0, target.y || 0, target.z || 0);
      
      // Edge color based on type
      const edgeType = link.edge_type || 'parent';
      const color = EDGE_COLORS[edgeType] || EDGE_COLORS.parent;
      
      // Add color for both vertices
      colors.push(color.r, color.g, color.b);
      colors.push(color.r, color.g, color.b);
    });
    
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    return geom;
  }, [visibleLinks, nodeMap]);
  
  if (visibleLinks.length === 0) return null;
  
  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={edgeOpacity}
        depthWrite={false}
      />
    </lineSegments>
  );
}

/**
 * Main scene component
 */
function Scene({
  nodes,
  links,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  enabledEdgeTypes,
  maxVisibleNodes = 500000,
  maxVisibleEdges = 100000,
  nodeSizeMultiplier = 1.0,
  edgeOpacity = 0.6,
  colorBy = 'library',
  sizeBy = 'downloads',
  colorScheme = 'viridis',
  highlightedNodeId,
  familyFilter,
  searchQuery,
}: ForceDirectedGraph3DInstancedProps) {
  return (
    <>
      <Edges
        nodes={nodes}
        links={links}
        enabledEdgeTypes={enabledEdgeTypes}
        maxVisible={maxVisibleEdges}
        edgeOpacity={edgeOpacity}
        familyFilter={familyFilter}
        searchQuery={searchQuery}
      />
      <InstancedNodes
        nodes={nodes}
        selectedNodeId={selectedNodeId}
        onNodeClick={onNodeClick}
        onNodeHover={onNodeHover}
        maxVisible={maxVisibleNodes}
        nodeSizeMultiplier={nodeSizeMultiplier}
        colorBy={colorBy}
        sizeBy={sizeBy}
        colorScheme={colorScheme}
        highlightedNodeId={highlightedNodeId}
        familyFilter={familyFilter}
        searchQuery={searchQuery}
      />
    </>
  );
}

/**
 * Main component with Canvas wrapper
 */
export default function ForceDirectedGraph3DInstanced({
  width,
  height,
  nodes,
  links,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  enabledEdgeTypes,
  showLabels = false,
  maxVisibleNodes = 500000,
  maxVisibleEdges = 100000,
  linkDistance = 100,
  chargeStrength = -300,
  collisionRadius = 1.0,
  nodeSizeMultiplier = 1.0,
  edgeOpacity = 0.6,
  colorBy = 'library',
  sizeBy = 'downloads',
  colorScheme = 'viridis',
  highlightedNodeId,
  familyFilter,
  searchQuery,
}: ForceDirectedGraph3DInstancedProps) {
  // Calculate bounds for camera positioning
  const bounds = useMemo(() => {
    if (nodes.length === 0) {
      return { center: [0, 0, 0] as [number, number, number], radius: 100 };
    }
    
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    // Sample nodes for bounds calculation if too many
    const sampleNodes = nodes.length > 10000 
      ? nodes.filter((_, i) => i % Math.ceil(nodes.length / 10000) === 0)
      : nodes;
    
    sampleNodes.forEach(node => {
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
  
  if (nodes.length === 0) {
    return (
      <div className="force-directed-graph-container">
        <div className="graph-empty">No nodes to display</div>
      </div>
    );
  }
  
  return (
    <div className="force-directed-graph-container" style={{ width, height }}>
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          stencil: false,
          depth: true,
        }}
        camera={{
          position: [
            bounds.center[0] + bounds.radius * 1.5,
            bounds.center[1] + bounds.radius * 1.5,
            bounds.center[2] + bounds.radius * 1.5,
          ],
          fov: 45,
          near: 0.1,
          far: bounds.radius * 20,
        }}
        frameloop="demand"
      >
        <color attach="background" args={['#1a1a1a']} />
        
        <OrbitControls
          target={bounds.center}
          enableDamping={true}
          dampingFactor={0.05}
          minDistance={bounds.radius * 0.1}
          maxDistance={bounds.radius * 5}
          makeDefault
        />
        
        <ambientLight intensity={0.8} />
        <directionalLight position={[1, 1, 1]} intensity={0.5} />
        
        <Scene
          nodes={nodes}
          links={links}
          onNodeClick={onNodeClick}
          onNodeHover={onNodeHover}
          selectedNodeId={selectedNodeId}
          enabledEdgeTypes={enabledEdgeTypes}
          maxVisibleNodes={maxVisibleNodes}
          maxVisibleEdges={maxVisibleEdges}
          width={width}
          height={height}
          nodeSizeMultiplier={nodeSizeMultiplier}
          edgeOpacity={edgeOpacity}
          colorBy={colorBy}
          sizeBy={sizeBy}
          colorScheme={colorScheme}
          highlightedNodeId={highlightedNodeId}
          familyFilter={familyFilter}
          searchQuery={searchQuery}
        />
      </Canvas>
      
      {/* Performance info overlay */}
      <div className="graph-performance-info">
        <div>Nodes: {nodes.length.toLocaleString()}</div>
        <div>Edges: {links.length.toLocaleString()}</div>
        {nodes.length > maxVisibleNodes && (
          <div className="graph-performance-warning">
            Showing top {maxVisibleNodes.toLocaleString()} by popularity
          </div>
        )}
      </div>
    </div>
  );
}



