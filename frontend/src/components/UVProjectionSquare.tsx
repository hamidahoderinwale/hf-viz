/**
 * UV Parameterization Square - 2D projection of latent space for navigation.
 * Acts as a mini-map/control panel to navigate the 3D space.
 */
import React, { useMemo, useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../types';
import { getCategoricalColorMap, getContinuousColorScale } from '../utils/colors';

interface UVProjectionSquareProps {
  width: number;
  height: number;
  data: ModelPoint[];
  familyTree?: ModelPoint[];
  colorBy: string;
  onRegionSelect?: (center: { x: number; y: number; z: number }) => void;
  selectedModelId?: string | null;
  currentViewCenter?: { x: number; y: number; z: number } | null;
}

export default function UVProjectionSquare({
  width,
  height,
  data,
  familyTree,
  colorBy,
  onRegionSelect,
  selectedModelId,
  currentViewCenter,
}: UVProjectionSquareProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);

  // Sample data for very large datasets to improve performance and navigability
  const sampledData = useMemo(() => {
    // Reduced limit for better sparsity in minimap
    const renderLimit = 15000; // Reduced from 20K to 15K
    if (data.length <= renderLimit) return data;
    
    // Use step-based sampling for better distribution
    const step = Math.ceil(data.length / renderLimit);
    const sampled: typeof data = [];
    for (let i = 0; i < data.length; i += step) {
      sampled.push(data[i]);
    }
    return sampled;
  }, [data]);

  const { xScale, yScale, colorScale, sizeScale, familyMap } = useMemo(() => {
    if (sampledData.length === 0) {
      return {
        xScale: d3.scaleLinear().domain([0, 1]).range([0, width]),
        yScale: d3.scaleLinear().domain([0, 1]).range([height, 0]),
        colorScale: () => '#808080',
        sizeScale: () => 2,
        familyMap: new Map<string, ModelPoint>(),
      };
    }

    const xExtent = [Math.min(...sampledData.map(d => d.x)), Math.max(...sampledData.map(d => d.x))] as [number, number];
    const yExtent = [Math.min(...sampledData.map(d => d.y)), Math.max(...sampledData.map(d => d.y))] as [number, number];

    const padding = 20;
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([padding, width - padding]);

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([height - padding, padding]);

    // Color scale with improved color schemes
    const isCategorical = colorBy === 'library_name' || colorBy === 'pipeline_tag';
    let colorScale: (d: ModelPoint) => string;

    if (isCategorical) {
      const categories = Array.from(new Set(sampledData.map(d => {
        if (colorBy === 'library_name') return d.library_name || 'unknown';
        return d.pipeline_tag || 'unknown';
      })));
      const colorScheme = colorBy === 'library_name' ? 'library' : 'pipeline';
      const colorMap = getCategoricalColorMap(categories, colorScheme);
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'library_name' ? d.library_name : d.pipeline_tag;
        return colorMap.get(val || 'unknown') || '#808080';
      };
    } else {
      const values = sampledData.map(d => colorBy === 'downloads' ? d.downloads : d.likes);
      const min = Math.min(...values);
      const max = Math.max(...values);
      // Use logarithmic scaling for downloads/likes (heavily skewed distributions)
      const useLogScale = colorBy === 'downloads' || colorBy === 'likes';
      const continuousScale = getContinuousColorScale(min, max, 'viridis', useLogScale);
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'downloads' ? d.downloads : d.likes;
        return continuousScale(val);
      };
    }

    // Size scale with logarithmic scaling for better representation
    const sizeValues = sampledData.map(d => {
      if (colorBy === 'downloads') return d.downloads;
      if (colorBy === 'likes') return d.likes;
      return 1;
    });
    const sizeMin = Math.min(...sizeValues);
    const sizeMax = Math.max(...sizeValues);
    // Use logarithmic scaling for downloads/likes
    const useLogSize = colorBy === 'downloads' || colorBy === 'likes';
    const logSizeMin = useLogSize && sizeMin > 0 ? Math.log10(sizeMin + 1) : sizeMin;
    const logSizeMax = useLogSize && sizeMax > 0 ? Math.log10(sizeMax + 1) : sizeMax;
    const logSizeRange = logSizeMax - logSizeMin || 1;
    const sizeRange = sizeMax - sizeMin || 1;
    const sizeScale = (d: ModelPoint) => {
      let normalizedSize: number;
      const val = colorBy === 'downloads' ? d.downloads : colorBy === 'likes' ? d.likes : 1;
      if (useLogSize && val > 0) {
        const logVal = Math.log10(val + 1);
        normalizedSize = (logVal - logSizeMin) / logSizeRange;
      } else {
        normalizedSize = (val - sizeMin) / sizeRange;
      }
      return 1 + Math.max(0, Math.min(1, normalizedSize)) * 2; // 1 to 3 pixel radius
    };

    // Family map
    const familyMap = new Map<string, ModelPoint>();
    if (familyTree) {
      familyTree.forEach(model => {
        familyMap.set(model.model_id, model);
      });
    }

    return { xScale, yScale, colorScale, sizeScale, familyMap };
  }, [sampledData, familyTree, colorBy, width, height]);

  // Build family edges for 2D projection
  const familyEdges = useMemo(() => {
    if (!familyTree || familyTree.length === 0) return [];
    
    const edges: Array<{ start: ModelPoint; end: ModelPoint }> = [];
    const modelMap = new Map(familyTree.map(m => [m.model_id, m]));

    familyTree.forEach(model => {
      if (model.parent_model && modelMap.has(model.parent_model)) {
        const parent = modelMap.get(model.parent_model)!;
        edges.push({ start: parent, end: model });
      }
    });

    return edges;
  }, [familyTree]);

  useEffect(() => {
    if (!svgRef.current || sampledData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Draw family tree edges
    if (familyEdges.length > 0) {
      g.append('g')
        .attr('class', 'family-edges')
        .selectAll('line')
        .data(familyEdges)
        .enter()
        .append('line')
        .attr('x1', d => xScale(d.start.x))
        .attr('y1', d => yScale(d.start.y))
        .attr('x2', d => xScale(d.end.x))
        .attr('y2', d => yScale(d.end.y))
        .attr('stroke', '#8a8a8a')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.4);
    }

    // Draw points
    const points = g
      .append('g')
      .attr('class', 'points')
      .selectAll<SVGCircleElement, ModelPoint>('circle')
      .data(sampledData)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => {
        const isFamilyMember = familyMap.has(d.model_id);
        const isSelected = selectedModelId === d.model_id;
        const baseSize = sizeScale(d);
        return isSelected ? baseSize * 2 : isFamilyMember ? baseSize * 1.5 : baseSize;
      })
      .attr('fill', d => {
        const isFamilyMember = familyMap.has(d.model_id);
        const isSelected = selectedModelId === d.model_id;
        if (isSelected) return '#ffffff';
        if (isFamilyMember) return '#4a4a4a';
        return colorScale(d);
      })
      .attr('stroke', d => {
        const isSelected = selectedModelId === d.model_id;
        const isFamilyMember = familyMap.has(d.model_id);
        if (isSelected) return '#000000';
        if (isFamilyMember) return '#6a6a6a';
        return hoveredPoint === d.model_id ? '#4a4a4a' : '#ffffff';
      })
      .attr('stroke-width', d => {
        const isSelected = selectedModelId === d.model_id;
        const isFamilyMember = familyMap.has(d.model_id);
        if (isSelected) return 2;
        if (isFamilyMember) return 1.5;
        return hoveredPoint === d.model_id ? 1.5 : 0.5;
      })
      .style('cursor', 'pointer')
      .style('opacity', d => {
        const isFamilyMember = familyMap.has(d.model_id);
        const isSelected = selectedModelId === d.model_id;
        if (isSelected || isFamilyMember) return 1;
        return hoveredPoint === d.model_id ? 0.9 : 0.7;
      })
      .on('click', function (event, d) {
        event.stopPropagation();
        if (onRegionSelect) {
          // Find the z coordinate for this point
          const zValue = d.z;
          onRegionSelect({ x: d.x, y: d.y, z: zValue });
        }
      })
      .on('mouseover', function (event, d) {
        setHoveredPoint(d.model_id);
        d3.select(this)
          .attr('r', sizeScale(d) * 1.5)
          .style('opacity', 1);
      })
      .on('mouseout', function (event, d) {
        setHoveredPoint(null);
        const isFamilyMember = familyMap.has(d.model_id);
        const isSelected = selectedModelId === d.model_id;
        d3.select(this)
          .attr('r', isSelected ? sizeScale(d) * 2 : isFamilyMember ? sizeScale(d) * 1.5 : sizeScale(d))
          .style('opacity', isSelected || isFamilyMember ? 1 : 0.7);
      });

    // Draw current view center indicator
    if (currentViewCenter) {
      g.append('circle')
        .attr('cx', xScale(currentViewCenter.x))
        .attr('cy', yScale(currentViewCenter.y))
        .attr('r', 8)
        .attr('fill', 'none')
        .attr('stroke', '#ff0000')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4,4')
        .style('pointer-events', 'none');
    }

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 20})`)
      .call(xAxis)
      .style('font-size', '10px')
      .style('color', '#666');

    g.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(20, 0)`)
      .call(yAxis)
      .style('font-size', '10px')
      .style('color', '#666');

    // Add title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .style('fill', '#333')
      .text('UV Projection (XY Plane)');

  }, [sampledData, familyTree, familyEdges, xScale, yScale, colorScale, sizeScale, familyMap, selectedModelId, hoveredPoint, onRegionSelect, currentViewCenter, width, height]);

  return (
    <div style={{ width, height, background: '#ffffff', border: '1px solid #d0d0d0', borderRadius: '2px' }}>
      <svg ref={svgRef} width={width} height={height} style={{ display: 'block' }} />
      <div style={{
        position: 'absolute',
        bottom: 5,
        right: 5,
        fontSize: '10px',
        color: '#666',
        background: 'rgba(255, 255, 255, 0.9)',
        padding: '2px 6px',
        borderRadius: '2px',
        fontFamily: "'Vend Sans', sans-serif",
      }}>
        Click to navigate
      </div>
    </div>
  );
}



