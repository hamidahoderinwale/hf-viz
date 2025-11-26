/**
 * Heatmap view showing density of models in latent space.
 * Uses D3.js for rendering.
 */
import React, { useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';
import './HeatmapView.css';

interface HeatmapViewProps {
  data: ModelPoint[];
  width?: number;
  height?: number;
}

export default function HeatmapView({ data, width = 800, height = 600 }: HeatmapViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const heatmapData = useMemo(() => {
    if (data.length === 0) return null;

    const gridSize = 50;
    const grid: number[][] = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));

    // Find bounds
    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Populate grid
    data.forEach(model => {
      const x = Math.floor(((model.x - xMin) / (xMax - xMin)) * (gridSize - 1));
      const y = Math.floor(((model.y - yMin) / (yMax - yMin)) * (gridSize - 1));
      if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
        grid[y][x]++;
      }
    });

    return { grid, xMin, xMax, yMin, yMax };
  }, [data]);

  useEffect(() => {
    if (!heatmapData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const gridSize = heatmapData.grid.length;
    const cellWidth = innerWidth / gridSize;
    const cellHeight = innerHeight / gridSize;

    const maxValue = Math.max(...heatmapData.grid.flat());
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, maxValue]);

    // Create heatmap cells
    heatmapData.grid.forEach((row, y) => {
      row.forEach((value, x) => {
        g.append('rect')
          .attr('x', x * cellWidth)
          .attr('y', y * cellHeight)
          .attr('width', cellWidth)
          .attr('height', cellHeight)
          .attr('fill', colorScale(value))
          .attr('stroke', 'none')
          .append('title')
          .text(`Density: ${value} models`);
      });
    });

    // Add axes
    const xScale = d3.scaleLinear()
      .domain([heatmapData.xMin, heatmapData.xMax])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([heatmapData.yMin, heatmapData.yMax])
      .range([innerHeight, 0]);

    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .attr('fill', 'var(--text-primary, #1a1a1a)')
      .style('text-anchor', 'middle')
      .text('X Coordinate');

    g.append('g')
      .call(yAxis)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -30)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'var(--text-primary, #1a1a1a)')
      .style('text-anchor', 'middle')
      .text('Y Coordinate');

    // Add color legend
    const legendWidth = 20;
    const legendHeight = 200;
    const legendX = innerWidth + 10;

    const legendScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([legendHeight, 0]);

    const legendAxis = d3.axisRight(legendScale).ticks(5);

    const legendG = g.append('g')
      .attr('transform', `translate(${legendX}, 0)`);

    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '0%')
      .attr('y2', '100%');

    const numStops = 10;
    for (let i = 0; i <= numStops; i++) {
      const value = (i / numStops) * maxValue;
      gradient.append('stop')
        .attr('offset', `${(i / numStops) * 100}%`)
        .attr('stop-color', colorScale(value));
    }

    legendG.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#heatmap-gradient)');

    legendG.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(legendAxis);

  }, [heatmapData, width, height]);

  if (!heatmapData) {
    return (
      <div className="heatmap-view">
        <div className="heatmap-empty">No data to display</div>
      </div>
    );
  }

  return (
    <div className="heatmap-view">
      <div className="heatmap-header">
        <h3>Model Density Heatmap</h3>
        <div className="heatmap-stats">Total Models: {data.length.toLocaleString()}</div>
      </div>
      <svg ref={svgRef} width={width} height={height} className="heatmap-svg" />
    </div>
  );
}

