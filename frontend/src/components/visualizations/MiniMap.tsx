import React, { useRef, useEffect, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';

interface MiniMapProps {
  width: number;
  height: number;
  data: ModelPoint[];
  colorBy: string;
  // Main plot dimensions
  mainWidth: number;
  mainHeight: number;
  mainMargin: { top: number; right: number; bottom: number; left: number };
  // Current transform from main plot
  transform: d3.ZoomTransform;
  // Callback to update main plot transform
  onViewportChange?: (transform: d3.ZoomTransform) => void;
  // Base scales from main plot
  xScaleBase: d3.ScaleLinear<number, number>;
  yScaleBase: d3.ScaleLinear<number, number>;
}

export default function MiniMap({
  width,
  height,
  data,
  colorBy,
  mainWidth,
  mainHeight,
  mainMargin,
  transform,
  onViewportChange,
  xScaleBase,
  yScaleBase,
}: MiniMapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const isDragging = useRef(false);

  // Sample data for mini-map (show fewer points for performance)
  const sampledData = useMemo(() => {
    const maxPoints = 2000;
    if (data.length <= maxPoints) return data;
    
    const step = Math.ceil(data.length / maxPoints);
    return data.filter((_, i) => i % step === 0);
  }, [data]);

  // Mini-map scales (fit entire data in mini-map viewport)
  const { miniXScale, miniYScale, colorScale } = useMemo(() => {
    if (data.length === 0) {
      return {
        miniXScale: d3.scaleLinear(),
        miniYScale: d3.scaleLinear(),
        colorScale: () => '#666',
      };
    }

    const padding = 8;
    const xExtent = d3.extent(data, (d) => d.x) as [number, number];
    const yExtent = d3.extent(data, (d) => d.y) as [number, number];

    const miniXScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([padding, width - padding]);

    const miniYScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([height - padding, padding]);

    // Color scale
    let colorScale: (d: ModelPoint) => string;
    
    if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
      const categories = Array.from(new Set(data.map((d) => 
        colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown')
      )));
      const ordinalScale = d3.scaleOrdinal(d3.schemeTableau10).domain(categories);
      colorScale = (d: ModelPoint) => {
        const value = colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown');
        return ordinalScale(value);
      };
    } else {
      const values = data.map((d) => colorBy === 'downloads' ? d.downloads : d.likes);
      const extent = d3.extent(values) as [number, number];
      const logExtent: [number, number] = [Math.log10(extent[0] + 1), Math.log10(extent[1] + 1)];
      const sequentialScale = d3.scaleSequential(d3.interpolateTurbo).domain(logExtent);
      
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'downloads' ? d.downloads : d.likes;
        return sequentialScale(Math.log10(val + 1));
      };
    }

    return { miniXScale, miniYScale, colorScale };
  }, [data, width, height, colorBy]);

  // Calculate viewport rectangle in mini-map coordinates
  const viewportRect = useMemo(() => {
    if (!xScaleBase.domain || !yScaleBase.domain) return null;

    const mainPlotWidth = mainWidth - mainMargin.left - mainMargin.right;
    const mainPlotHeight = mainHeight - mainMargin.top - mainMargin.bottom;

    // Get the visible domain in data coordinates
    const visibleXDomain = transform.rescaleX(xScaleBase).domain();
    const visibleYDomain = transform.rescaleY(yScaleBase).domain();

    // Convert to mini-map coordinates
    const x = miniXScale(visibleXDomain[0]);
    const y = miniYScale(visibleYDomain[1]); // Note: y is inverted
    const rectWidth = miniXScale(visibleXDomain[1]) - miniXScale(visibleXDomain[0]);
    const rectHeight = miniYScale(visibleYDomain[0]) - miniYScale(visibleYDomain[1]);

    return {
      x: Math.max(0, x),
      y: Math.max(0, y),
      width: Math.min(width, rectWidth),
      height: Math.min(height, rectHeight),
    };
  }, [transform, xScaleBase, yScaleBase, miniXScale, miniYScale, mainWidth, mainHeight, mainMargin, width, height]);

  // Handle click on mini-map to pan
  const handleClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!onViewportChange || !svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Convert click position to data coordinates
    const dataX = miniXScale.invert(clickX);
    const dataY = miniYScale.invert(clickY);

    // Calculate new transform to center on clicked point
    const mainPlotWidth = mainWidth - mainMargin.left - mainMargin.right;
    const mainPlotHeight = mainHeight - mainMargin.top - mainMargin.bottom;

    // Get center of current viewport in data coordinates
    const newCenterX = xScaleBase(dataX);
    const newCenterY = yScaleBase(dataY);

    // Calculate translation to center on this point
    const newX = mainPlotWidth / 2 - transform.k * newCenterX;
    const newY = mainPlotHeight / 2 - transform.k * newCenterY;

    const newTransform = d3.zoomIdentity
      .translate(newX, newY)
      .scale(transform.k);

    onViewportChange(newTransform);
  }, [onViewportChange, miniXScale, miniYScale, xScaleBase, yScaleBase, mainWidth, mainHeight, mainMargin, transform.k]);

  // Handle drag on viewport
  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    event.stopPropagation();
    isDragging.current = true;
    document.body.style.cursor = 'grabbing';
  }, []);

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!isDragging.current || !onViewportChange || !svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Convert mouse position to data coordinates
    const dataX = miniXScale.invert(mouseX);
    const dataY = miniYScale.invert(mouseY);

    // Calculate new transform
    const mainPlotWidth = mainWidth - mainMargin.left - mainMargin.right;
    const mainPlotHeight = mainHeight - mainMargin.top - mainMargin.bottom;

    const newCenterX = xScaleBase(dataX);
    const newCenterY = yScaleBase(dataY);

    const newX = mainPlotWidth / 2 - transform.k * newCenterX;
    const newY = mainPlotHeight / 2 - transform.k * newCenterY;

    const newTransform = d3.zoomIdentity
      .translate(newX, newY)
      .scale(transform.k);

    onViewportChange(newTransform);
  }, [onViewportChange, miniXScale, miniYScale, xScaleBase, yScaleBase, mainWidth, mainHeight, mainMargin, transform.k]);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
    document.body.style.cursor = '';
  }, []);

  // Add global mouse event listeners for dragging
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  // Render mini-map
  useEffect(() => {
    if (!svgRef.current || sampledData.length === 0) return;

    const svg = d3.select(svgRef.current);
    
    // Only update points layer, preserve viewport rect
    let pointsGroup = svg.select<SVGGElement>('.minimap-points');
    if (pointsGroup.empty()) {
      svg.selectAll('*').remove();
      
      // Background
      svg.append('rect')
        .attr('class', 'minimap-bg')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'var(--bg-secondary, #1a1a1a)')
        .attr('rx', 4);

      // Points group
      pointsGroup = svg.append('g').attr('class', 'minimap-points');
    }

    // Draw points
    pointsGroup
      .selectAll<SVGCircleElement, ModelPoint>('circle')
      .data(sampledData, (d) => d.model_id)
      .join(
        (enter) => enter
          .append('circle')
          .attr('cx', (d) => miniXScale(d.x))
          .attr('cy', (d) => miniYScale(d.y))
          .attr('r', 1.5)
          .attr('fill', (d) => colorScale(d))
          .attr('opacity', 0.6),
        (update) => update
          .attr('cx', (d) => miniXScale(d.x))
          .attr('cy', (d) => miniYScale(d.y))
          .attr('fill', (d) => colorScale(d)),
        (exit) => exit.remove()
      );

  }, [sampledData, width, height, miniXScale, miniYScale, colorScale]);

  // Update viewport rectangle separately for performance
  useEffect(() => {
    if (!svgRef.current || !viewportRect) return;

    const svg = d3.select(svgRef.current);
    
    // Remove old viewport
    svg.selectAll('.viewport-rect, .viewport-border').remove();

    // Viewport fill
    svg.append('rect')
      .attr('class', 'viewport-rect')
      .attr('x', viewportRect.x)
      .attr('y', viewportRect.y)
      .attr('width', Math.max(viewportRect.width, 10))
      .attr('height', Math.max(viewportRect.height, 10))
      .attr('fill', 'rgba(74, 144, 226, 0.15)')
      .attr('stroke', 'rgba(74, 144, 226, 0.8)')
      .attr('stroke-width', 2)
      .attr('rx', 2)
      .style('cursor', 'grab')
      .style('pointer-events', 'all');

    // Corner handles for visual feedback
    const handleSize = 6;
    const corners = [
      { x: viewportRect.x, y: viewportRect.y },
      { x: viewportRect.x + viewportRect.width, y: viewportRect.y },
      { x: viewportRect.x, y: viewportRect.y + viewportRect.height },
      { x: viewportRect.x + viewportRect.width, y: viewportRect.y + viewportRect.height },
    ];

    svg.selectAll('.viewport-handle')
      .data(corners)
      .join('rect')
      .attr('class', 'viewport-handle')
      .attr('x', (d) => d.x - handleSize / 2)
      .attr('y', (d) => d.y - handleSize / 2)
      .attr('width', handleSize)
      .attr('height', handleSize)
      .attr('fill', 'rgba(74, 144, 226, 1)')
      .attr('rx', 1);

  }, [viewportRect]);

  if (data.length === 0) return null;

  return (
    <div className="minimap-container">
      <div className="minimap-header">
        <span className="minimap-title">Overview Map</span>
        <span className="minimap-hint">Click to navigate</span>
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="minimap-svg"
        onClick={handleClick}
        onMouseDown={handleMouseDown}
      />
      <div className="minimap-stats">
        <span>Zoom: {transform.k.toFixed(1)}x</span>
      </div>
    </div>
  );
}

