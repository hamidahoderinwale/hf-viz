/**
 * D3.js network graph showing model family relationships and connectivity.
 * Uses Web Worker for efficient distance calculations.
 */
import React, { useMemo, useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';

interface NetworkGraphProps {
  width: number;
  height: number;
  data: ModelPoint[];
  onNodeClick?: (model: ModelPoint) => void;
}

interface Node extends d3.SimulationNodeDatum {
  id: string;
  model: ModelPoint;
  group?: string;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: Node | string;
  target: Node | string;
}

export default function NetworkGraph({
  width,
  height,
  data,
  onNodeClick,
}: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const [links, setLinks] = useState<Link[]>([]);
  const [isCalculating, setIsCalculating] = useState(false);

  // Create nodes from models
  const nodes = useMemo(() => {
    const nodeMap = new Map<string, Node>();
    data.forEach((model) => {
      if (!nodeMap.has(model.model_id)) {
        nodeMap.set(model.model_id, {
          id: model.model_id,
          model,
          group: model.library_name || 'unknown',
        });
      }
    });
    return Array.from(nodeMap.values());
  }, [data]);

  // Initialize Web Worker
  useEffect(() => {
    // Create worker from inline code (for webpack compatibility)
    const workerCode = `
      // Spatial grid for efficient nearest neighbor search
      class SpatialGrid {
        constructor(data, cellSize = 0.15) {
          this.grid = new Map();
          this.cellSize = cellSize;
          data.forEach((point) => {
            const cellKey = this.getCellKey(point.x, point.y);
            if (!this.grid.has(cellKey)) {
              this.grid.set(cellKey, []);
            }
            this.grid.get(cellKey).push(point);
          });
        }

        getCellKey(x, y) {
          const cellX = Math.floor(x / this.cellSize);
          const cellY = Math.floor(y / this.cellSize);
          return \`\${cellX},\${cellY}\`;
        }

        findNeighbors(point, threshold) {
          const neighbors = [];
          const cellX = Math.floor(point.x / this.cellSize);
          const cellY = Math.floor(point.y / this.cellSize);

          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              const cellKey = \`\${cellX + dx},\${cellY + dy}\`;
              const cellPoints = this.grid.get(cellKey);
              if (cellPoints) {
                cellPoints.forEach((p) => {
                  if (p.model_id !== point.model_id) {
                    const dx2 = p.x - point.x;
                    const dy2 = p.y - point.y;
                    const distance = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                    if (distance < threshold) {
                      neighbors.push({ ...p, distance });
                    }
                  }
                });
              }
            }
          }
          return neighbors;
        }
      }

      function calculateDistances(data, threshold, maxLinks, maxNodes) {
        const links = [];
        const linkSet = new Set();
        const nodesToProcess = maxNodes ? data.slice(0, maxNodes) : data;
        const spatialGrid = new SpatialGrid(data, threshold * 2);

        nodesToProcess.forEach((model) => {
          if (links.length >= maxLinks) return;
          const neighbors = spatialGrid.findNeighbors(model, threshold);
          neighbors
            .sort((a, b) => a.distance - b.distance)
            .slice(0, 2)
            .forEach((neighbor) => {
              const linkKey = [model.model_id, neighbor.model_id].sort().join('-');
              if (!linkSet.has(linkKey) && links.length < maxLinks) {
                linkSet.add(linkKey);
                links.push({
                  source: model.model_id,
                  target: neighbor.model_id,
                  distance: neighbor.distance,
                });
              }
            });
        });

        return { links };
      }

      self.onmessage = (e) => {
        const { type, data, threshold, maxLinks, maxNodes } = e.data;
        if (type === 'calculateDistances') {
          try {
            const result = calculateDistances(data, threshold, maxLinks, maxNodes);
            self.postMessage({ type: 'result', result });
          } catch (error) {
            self.postMessage({
              type: 'error',
              error: error.message || 'Unknown error',
            });
          }
        }
      };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    workerRef.current = new Worker(URL.createObjectURL(blob));

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  // Calculate links using Web Worker
  useEffect(() => {
    if (!workerRef.current || data.length === 0) {
      setLinks([]);
      return;
    }

    setIsCalculating(true);
    const threshold = 0.15;
    // Scale maxLinks with data size, but cap for performance
    const maxLinks = Math.min(data.length * 2, 2000);
    // For very large datasets, limit nodes processed but use spatial indexing
    const maxNodes = data.length > 1000 ? Math.min(1000, data.length) : undefined;

    const worker = workerRef.current;
    worker.onmessage = (e) => {
      if (e.data.type === 'result') {
        setLinks(e.data.result.links);
        setIsCalculating(false);
      } else if (e.data.type === 'error') {
        console.error('Worker error:', e.data.error);
        setIsCalculating(false);
      }
    };

    worker.postMessage({
      type: 'calculateDistances',
      data,
      threshold,
      maxLinks,
      maxNodes,
    });

    return () => {
      worker.onmessage = null;
    };
  }, [data]);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;
    
    // Don't render if still calculating and no links yet
    if (isCalculating && links.length === 0) {
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('fill', '#666')
        .text('Calculating network connections...');
      return;
    }
    
    // Don't render if no links (but not calculating)
    if (!isCalculating && links.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Set up simulation
    const simulation = d3
      .forceSimulation<Node>(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(50)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(5));

    // Create container
    const g = svg.append('g');

    // Create links
    const link = g
      .append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', (d) => Math.sqrt(1));

    // Create nodes
    const node = g
      .append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d) => {
        const downloads = d.model.downloads || 0;
        return 3 + Math.sqrt(downloads) / 100;
      })
      .attr('fill', (d) => {
        const colors = d3.schemeCategory10;
        const groups = Array.from(new Set(nodes.map((n) => n.group))).sort();
        const groupIndex = groups.indexOf(d.group || 'unknown');
        return colors[groupIndex % colors.length];
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (onNodeClick) onNodeClick(d.model);
      })
      .call(drag(simulation) as any);

    // Add labels
    const label = g
      .append('g')
      .selectAll('text')
      .data(nodes.filter((d) => (d.model.downloads || 0) > 10000))
      .join('text')
      .text((d) => d.model.model_id.split('/').pop() || d.id)
      .attr('font-size', '10px')
      .attr('dx', 6)
      .attr('dy', 4);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as Node).x!)
        .attr('y1', (d) => (d.source as Node).y!)
        .attr('x2', (d) => (d.target as Node).x!)
        .attr('y2', (d) => (d.target as Node).y!);

      node.attr('cx', (d) => d.x!).attr('cy', (d) => d.y!);

      label.attr('x', (d) => d.x!).attr('y', (d) => d.y!);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, links, width, height, onNodeClick, isCalculating]);

  return <svg ref={svgRef} width={width} height={height} />;
}

function drag(simulation: d3.Simulation<Node, undefined>) {
  function dragstarted(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3
    .drag<SVGCircleElement, Node>()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended);
}

