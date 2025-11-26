/**
 * Stacked hierarchical view showing model families and relationships.
 * Uses D3.js for hierarchical layout.
 */
import React, { useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';
import './StackedView.css';

interface StackedViewProps {
  data: ModelPoint[];
  width?: number;
  height?: number;
}

interface HierarchyNode {
  name: string;
  children?: HierarchyNode[];
  value?: number;
  models?: ModelPoint[];
}

export default function StackedView({ data, width = 800, height = 600 }: StackedViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const hierarchyData = useMemo(() => {
    if (data.length === 0) return null;

    // Build hierarchy: Library → Pipeline → Family
    const libraryMap = new Map<string, Map<string, Map<string, ModelPoint[]>>>();

    data.forEach(model => {
      const library = model.library_name || 'Unknown';
      const pipeline = model.pipeline_tag || 'Unknown';
      const family = model.parent_model || 'Root';

      if (!libraryMap.has(library)) {
        libraryMap.set(library, new Map());
      }
      const pipelineMap = libraryMap.get(library)!;

      if (!pipelineMap.has(pipeline)) {
        pipelineMap.set(pipeline, new Map());
      }
      const familyMap = pipelineMap.get(pipeline)!;

      if (!familyMap.has(family)) {
        familyMap.set(family, []);
      }
      familyMap.get(family)!.push(model);
    });

    // Convert to hierarchy structure
    const root: HierarchyNode = {
      name: 'Models',
      children: Array.from(libraryMap.entries()).map(([library, pipelineMap]) => ({
        name: library,
        children: Array.from(pipelineMap.entries()).map(([pipeline, familyMap]) => ({
          name: pipeline,
          children: Array.from(familyMap.entries()).map(([family, models]) => ({
            name: family,
            value: models.length,
            models,
          })),
        })),
      })),
    };

    return root;
  }, [data]);

  useEffect(() => {
    if (!hierarchyData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const root = d3.hierarchy(hierarchyData);
    const treeLayout = d3.tree<HierarchyNode>().size([height - 100, width - 200]);
    treeLayout(root);

    const g = svg.append('g').attr('transform', 'translate(100, 50)');

    // Links
    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', (d: d3.HierarchyLink<HierarchyNode>) => {
        const link = d3.linkHorizontal<d3.HierarchyPointNode<HierarchyNode>, d3.HierarchyPointNode<HierarchyNode>>()
          .x((d: d3.HierarchyPointNode<HierarchyNode>) => d.y)
          .y((d: d3.HierarchyPointNode<HierarchyNode>) => d.x);
        return link(d as any);
      })
      .attr('fill', 'none')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 1.5);

    // Nodes
    const nodes = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.y},${d.x})`);

    nodes.append('circle')
      .attr('r', d => Math.sqrt(d.data.value || 1) * 3 + 4)
      .attr('fill', d => d.depth === 0 ? '#4a90e2' : d.depth === 1 ? '#6ba3e8' : '#8bb5ed')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    nodes.append('text')
      .attr('dy', '.35em')
      .attr('x', d => (d.children ? -13 : 13))
      .style('text-anchor', d => (d.children ? 'end' : 'start'))
      .text(d => d.data.name)
      .style('font-size', '10px')
      .style('fill', 'var(--text-primary, #1a1a1a)');

  }, [hierarchyData, width, height]);

  if (!hierarchyData) {
    return (
      <div className="stacked-view">
        <div className="stacked-empty">No data to display</div>
      </div>
    );
  }

  return (
    <div className="stacked-view">
      <div className="stacked-header">
        <h3>Hierarchical Model Structure</h3>
        <div className="stacked-stats">Total Models: {data.length.toLocaleString()}</div>
      </div>
      <svg ref={svgRef} width={width} height={height} className="stacked-svg" />
    </div>
  );
}

