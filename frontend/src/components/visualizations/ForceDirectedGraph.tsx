/**
 * Force-directed graph visualization showing model relationships.
 * Displays different types of derivatives (finetunes, adapters, quantizations, merges)
 * with color-coded edges and interactive nodes.
 */
import React, { useMemo, useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import './ForceDirectedGraph.css';

export type EdgeType = 'finetune' | 'quantized' | 'adapter' | 'merge' | 'parent';

export interface GraphNode {
  id: string;
  title: string;
  downloads: number;
  likes: number;
  library: string;
  pipeline: string;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  edge_type: EdgeType;
  edge_types?: EdgeType[];
  change_in_downloads?: number;
  change_in_likes?: number;
}

interface ProcessedLink {
  source: string;
  target: string;
  edge_type: EdgeType;
  edge_types?: EdgeType[];
  change_in_downloads?: number;
  change_in_likes?: number;
}

export interface ForceDirectedGraphProps {
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

export default function ForceDirectedGraph({
  width,
  height,
  nodes,
  links,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  enabledEdgeTypes,
  showLabels = true,
}: ForceDirectedGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Filter links based on enabled edge types and ensure source/target are node IDs
  const filteredLinks = useMemo((): ProcessedLink[] => {
    let filtered = links;
    if (enabledEdgeTypes && enabledEdgeTypes.size > 0) {
      filtered = links.filter(link => {
        const linkTypes = link.edge_types || [link.edge_type];
        return linkTypes.some(type => enabledEdgeTypes.has(type));
      });
    }
    // Ensure source and target are strings (node IDs)
    return filtered.map(link => ({
      ...link,
      source: typeof link.source === 'string' ? link.source : (link.source as GraphNode).id,
      target: typeof link.target === 'string' ? link.target : (link.target as GraphNode).id,
    }));
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

  useEffect(() => {
    if (!svgRef.current || filteredNodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container
    const g = svg.append('g');

    // Set up zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Set initial transform
    const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8);
    svg.call(zoom.transform as any, initialTransform);

    // Set up force simulation
    const simulation = d3
      .forceSimulation<GraphNode>(filteredNodes)
      .force(
        'link',
        d3
          .forceLink<GraphNode, ProcessedLink>(filteredLinks)
          .id((d) => d.id)
          .distance((d) => {
            // Adjust distance based on edge type
            const link = d as unknown as ProcessedLink;
            const edgeType = link.edge_type;
            switch (edgeType) {
              case 'merge':
                return 120; // Merges are more distinct
              case 'finetune':
                return 80;  // Fine-tunes are closer
              case 'quantized':
                return 60;   // Quantizations are very close
              case 'adapter':
                return 70;
              default:
                return 100;
            }
          })
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide<GraphNode>().radius((d) => {
        // Node size based on downloads
        const downloads = d.downloads || 0;
        return 5 + Math.sqrt(downloads) / 200;
      }));

    simulationRef.current = simulation;

    // Create arrow markers for directed edges
    const defs = svg.append('defs');
    Object.entries(EDGE_COLORS).forEach(([type, color]) => {
      defs
        .append('marker')
        .attr('id', `arrow-${type}`)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color);
    });

    // Create links
    const link = g
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredLinks)
      .join('line')
      .attr('stroke', (d) => {
        const edgeType = d.edge_type;
        return EDGE_COLORS[edgeType] || EDGE_COLORS.parent;
      })
      .attr('stroke-width', (d) => {
        const edgeType = d.edge_type;
        return EDGE_STROKE_WIDTH[edgeType] || 1;
      })
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', (d) => {
        const edgeType = d.edge_type;
        return `url(#arrow-${edgeType})`;
      })
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d: ProcessedLink) {
        const edgeType = d.edge_type;
        d3.select(this)
          .attr('stroke-opacity', 1)
          .attr('stroke-width', (EDGE_STROKE_WIDTH[edgeType] || 1) + 1);
      })
      .on('mouseleave', function(event, d: ProcessedLink) {
        const edgeType = d.edge_type;
        d3.select(this)
          .attr('stroke-opacity', 0.6)
          .attr('stroke-width', EDGE_STROKE_WIDTH[edgeType] || 1);
      });

    // Create nodes
    const node = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(filteredNodes)
      .join('circle')
      .attr('r', (d) => {
        const downloads = d.downloads || 0;
        return 3 + Math.sqrt(downloads) / 200;
      })
      .attr('fill', (d) => {
        // Color by library if available
        if (d.library) {
          const colors = d3.schemeCategory10;
          const libraries = Array.from(new Set(filteredNodes.map(n => n.library).filter(Boolean)));
          const libIndex = libraries.indexOf(d.library);
          return colors[libIndex % colors.length];
        }
        return '#6b7280';
      })
      .attr('stroke', (d) => {
        if (selectedNodeId === d.id) {
          return '#ef4444';
        }
        if (hoveredNodeId === d.id) {
          return '#fbbf24';
        }
        return '#fff';
      })
      .attr('stroke-width', (d) => {
        if (selectedNodeId === d.id) {
          return 3;
        }
        if (hoveredNodeId === d.id) {
          return 2;
        }
        return 1.5;
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        if (onNodeClick) onNodeClick(d);
      })
      .on('mouseenter', (event, d) => {
        setHoveredNodeId(d.id);
        if (onNodeHover) onNodeHover(d);
        d3.select(event.currentTarget as SVGCircleElement)
          .attr('stroke-width', selectedNodeId === d.id ? 3 : 2.5);
      })
      .on('mouseleave', (event, d) => {
        setHoveredNodeId(null);
        if (onNodeHover) onNodeHover(null);
        d3.select(event.currentTarget as SVGCircleElement)
          .attr('stroke-width', selectedNodeId === d.id ? 3 : 1.5);
      })
      .call(drag(simulation) as any);

    // Create labels
    if (showLabels) {
      g
        .append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(filteredNodes.filter((d) => {
          // Show labels for nodes with high downloads or selected/hovered nodes
          return (d.downloads || 0) > 10000 || selectedNodeId === d.id || hoveredNodeId === d.id;
        }))
        .join('text')
        .text((d) => d.title || d.id.split('/').pop() || d.id)
        .attr('font-size', '10px')
        .attr('dx', 8)
        .attr('dy', 4)
        .attr('fill', '#fff')
        .attr('stroke', '#000')
        .attr('stroke-width', '0.5px')
        .attr('paint-order', 'stroke')
        .style('pointer-events', 'none');
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => {
          const sourceId = (d as unknown as ProcessedLink).source;
          const source = filteredNodes.find(n => n.id === sourceId);
          return source?.x || 0;
        })
        .attr('y1', (d) => {
          const sourceId = (d as unknown as ProcessedLink).source;
          const source = filteredNodes.find(n => n.id === sourceId);
          return source?.y || 0;
        })
        .attr('x2', (d) => {
          const targetId = (d as unknown as ProcessedLink).target;
          const target = filteredNodes.find(n => n.id === targetId);
          return target?.x || 0;
        })
        .attr('y2', (d) => {
          const targetId = (d as unknown as ProcessedLink).target;
          const target = filteredNodes.find(n => n.id === targetId);
          return target?.y || 0;
        });

      node.attr('cx', (d) => d.x || 0).attr('cy', (d) => d.y || 0);

      if (showLabels) {
        const label = g.selectAll<SVGTextElement, GraphNode>('.labels text');
        label.attr('x', (d) => d.x || 0).attr('y', (d) => d.y || 0);
      }
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [filteredNodes, filteredLinks, width, height, onNodeClick, onNodeHover, selectedNodeId, hoveredNodeId, showLabels]);

  return (
    <div className="force-directed-graph-container">
      <svg ref={svgRef} width={width} height={height} className="force-directed-graph" />
    </div>
  );
}

function drag(simulation: d3.Simulation<GraphNode, undefined>) {
  function dragstarted(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3
    .drag<SVGCircleElement, GraphNode>()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended);
}
