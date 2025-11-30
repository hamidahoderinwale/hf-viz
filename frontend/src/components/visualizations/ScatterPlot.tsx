import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';
import MiniMap from './MiniMap';
import './ScatterPlot.css';
import './MiniMap.css';

interface ScatterPlotProps {
  width: number;
  height: number;
  data: ModelPoint[];
  colorBy: string;
  sizeBy: string;
  margin?: { top: number; right: number; bottom: number; left: number };
  onPointClick?: (model: ModelPoint) => void;
  onBrush?: (selected: ModelPoint[]) => void;
}

const defaultMargin = { top: 40, right: 40, bottom: 60, left: 60 };

export default function ScatterPlot({
  width,
  height,
  data,
  colorBy,
  sizeBy,
  margin = defaultMargin,
  onPointClick,
  onBrush,
}: ScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const tooltipRef = useRef<SVGGElement | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<Set<string>>(new Set());
  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const zoomRef = useRef<d3.ZoomBehavior<Element, unknown> | null>(null);

  // Performance-optimized sampling with Level of Detail (LOD)
  const sampledData = useMemo(() => {
    // Increased render limit to support full dataset (using Canvas for performance)
    const renderLimit = 150000;
    
    // LOD: Reduce further when zoomed out
    const lodFactor = transform.k < 1 ? 0.5 : 1; // Show 50% when zoomed out
    const effectiveLimit = Math.floor(renderLimit * lodFactor);
    
    if (data.length <= effectiveLimit) return data;
    
    // Stratified sampling by quartiles to ensure representative distribution
    const sorted = [...data].sort((a, b) => a.x - b.x);
    const quartileSize = Math.floor(data.length / 4);
    const samplesPerQuartile = Math.floor(effectiveLimit / 4);
    
    const sampled: ModelPoint[] = [];
    for (let q = 0; q < 4; q++) {
      const start = q * quartileSize;
      const end = q === 3 ? data.length : (q + 1) * quartileSize;
      const quartile = sorted.slice(start, end);
      const step = Math.ceil(quartile.length / samplesPerQuartile);
      
      for (let i = 0; i < quartile.length; i += step) {
        sampled.push(quartile[i]);
      }
    }
    
    return sampled;
  }, [data, transform.k]);

  // Memoized scales with improved color schemes
  const { xScaleBase, yScaleBase, colorScale, sizeScale } = useMemo(() => {
    if (sampledData.length === 0) {
      return {
        xScaleBase: d3.scaleLinear(),
        yScaleBase: d3.scaleLinear(),
        colorScale: () => '#999',
        sizeScale: () => 5,
      };
    }

    const xExtent = d3.extent(sampledData, (d) => d.x) as [number, number];
    const yExtent = d3.extent(sampledData, (d) => d.y) as [number, number];

    const xScaleBase = d3
      .scaleLinear()
      .domain(xExtent)
      .range([0, width - margin.left - margin.right])
      .nice();

    const yScaleBase = d3
      .scaleLinear()
      .domain(yExtent)
      .range([height - margin.top - margin.bottom, 0])
      .nice();

    // Improved color scale
    let colorScale: (d: ModelPoint) => string;
    
    if (colorBy === 'library_name' || colorBy === 'pipeline_tag') {
      const categories = Array.from(new Set(sampledData.map((d) => 
        colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown')
      )));
      const ordinalScale = d3.scaleOrdinal(d3.schemeTableau10).domain(categories);
      colorScale = (d: ModelPoint) => {
        const value = colorBy === 'library_name' ? (d.library_name || 'unknown') : (d.pipeline_tag || 'unknown');
        return ordinalScale(value);
      };
    } else {
      const values = sampledData.map((d) => colorBy === 'downloads' ? d.downloads : d.likes);
      const extent = d3.extent(values) as [number, number];
      const logExtent: [number, number] = [Math.log10(extent[0] + 1), Math.log10(extent[1] + 1)];
      const sequentialScale = d3.scaleSequential(d3.interpolateTurbo).domain(logExtent);
      
      colorScale = (d: ModelPoint) => {
        const val = colorBy === 'downloads' ? d.downloads : d.likes;
        return sequentialScale(Math.log10(val + 1));
      };
    }

    // Improved size scale with better range
    const sizeValues = sampledData.map((d) => {
      if (sizeBy === 'downloads') return d.downloads;
      if (sizeBy === 'likes') return d.likes;
      return 1;
    });
    const sizeExtent = d3.extent(sizeValues) as [number, number];
    
    let sizeScale: (d: ModelPoint) => number;
    if (sizeBy === 'downloads' || sizeBy === 'likes') {
      const logExtent: [number, number] = [
        Math.log10(sizeExtent[0] + 1),
        Math.log10(sizeExtent[1] + 1)
      ];
      const logScale = d3.scaleSqrt().domain(logExtent).range([2.5, 15]);
      sizeScale = (d: ModelPoint) => {
        const val = sizeBy === 'downloads' ? d.downloads : d.likes;
        return logScale(Math.log10(val + 1));
      };
    } else {
      sizeScale = () => 5;
    }

    return { xScaleBase, yScaleBase, colorScale, sizeScale };
  }, [sampledData, width, height, margin, colorBy, sizeBy]);

  // Transform scales with zoom
  const xScale = useMemo(() => transform.rescaleX(xScaleBase), [xScaleBase, transform]);
  const yScale = useMemo(() => transform.rescaleY(yScaleBase), [yScaleBase, transform]);

  // Reset zoom handler
  const resetZoom = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(750)
        .ease(d3.easeCubicInOut)
        .call(zoomRef.current.transform as any, d3.zoomIdentity);
    }
  }, []);

  // Handler for mini-map viewport changes
  const handleMiniMapViewportChange = useCallback((newTransform: d3.ZoomTransform) => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .ease(d3.easeCubicOut)
        .call(zoomRef.current.transform as any, newTransform);
    }
  }, []);

  // Debounced tooltip update
  const showTooltip = useCallback((d: ModelPoint, x: number, y: number) => {
    if (!gRef.current) return;
    
    const g = d3.select(gRef.current);
    g.selectAll('.tooltip').remove();
    
    const tooltip = g
      .append('g')
      .attr('class', 'tooltip')
      .attr('transform', `translate(${x + 15},${y - 15})`)
      .style('pointer-events', 'none');

    tooltipRef.current = tooltip.node();

    const padding = 12;
    const lineHeight = 18;
    const lines = [
      { text: d.model_id.length > 35 ? d.model_id.substring(0, 35) + '...' : d.model_id, bold: true },
      { text: `Library: ${d.library_name || 'N/A'}` },
      { text: `Pipeline: ${d.pipeline_tag || 'N/A'}` },
      { text: `Downloads: ${d.downloads.toLocaleString()} | Likes: ${d.likes.toLocaleString()}` },
      { text: 'Click for details', small: true }
    ];

    const bgHeight = padding * 2 + lines.length * lineHeight;

    tooltip
      .append('rect')
      .attr('width', 240)
      .attr('height', bgHeight)
      .attr('fill', 'rgba(15, 15, 15, 0.96)')
      .attr('stroke', 'rgba(255, 255, 255, 0.2)')
      .attr('stroke-width', 1)
      .attr('rx', 6)
      .attr('filter', 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3))');

    lines.forEach((line, i) => {
      tooltip
        .append('text')
        .attr('x', padding)
        .attr('y', padding + (i + 1) * lineHeight)
        .attr('fill', line.bold ? 'white' : line.small ? '#999' : '#ccc')
        .attr('font-size', line.small ? '10px' : '12px')
        .attr('font-weight', line.bold ? 'bold' : 'normal')
        .text(line.text);
    });
  }, []);

  const hideTooltip = useCallback(() => {
    if (!gRef.current) return;
    d3.select(gRef.current).selectAll('.tooltip').remove();
    tooltipRef.current = null;
  }, []);

  useEffect(() => {
    if (!svgRef.current || sampledData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Zoom controls
    const controls = svg
      .append('g')
      .attr('class', 'zoom-controls')
      .attr('transform', `translate(${width - 130}, 20)`);

    controls
      .append('rect')
      .attr('width', 110)
      .attr('height', 130)
      .attr('fill', 'rgba(255, 255, 255, 0.97)')
      .attr('stroke', '#ddd')
      .attr('stroke-width', 1)
      .attr('rx', 6)
      .attr('filter', 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))');

    controls
      .append('text')
      .attr('x', 55)
      .attr('y', 22)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('fill', '#333')
      .text('ZOOM CONTROLS');

    const createButton = (label: string, yOffset: number, onClick: () => void) => {
      const btn = controls
        .append('g')
        .attr('transform', `translate(15, ${yOffset})`)
        .style('cursor', 'pointer')
        .on('click', onClick);

      btn
        .append('rect')
        .attr('width', 80)
        .attr('height', 32)
        .attr('fill', '#4a90e2')
        .attr('rx', 4)
        .on('mouseover', function() {
          d3.select(this).attr('fill', '#357abd');
        })
        .on('mouseout', function() {
          d3.select(this).attr('fill', '#4a90e2');
        });

      btn
        .append('text')
        .attr('x', 40)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('font-size', label === '+' || label === '−' ? '20px' : '13px')
        .attr('font-weight', label === '+' || label === '−' ? 'bold' : '600')
        .text(label);
    };

    createButton('+', 40, () => {
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current)
          .transition()
          .duration(300)
          .call(zoomRef.current.scaleBy as any, 1.5);
      }
    });

    createButton('−', 80, () => {
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current)
          .transition()
          .duration(300)
          .call(zoomRef.current.scaleBy as any, 1 / 1.5);
      }
    });

    createButton('Reset', 120, resetZoom);

    const g = svg
      .append('g')
      .attr('class', 'main-group')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    gRef.current = g.node() as SVGGElement;

    const xMax = width - margin.left - margin.right;
    const yMax = height - margin.top - margin.bottom;

    // Clip path for points
    svg
      .append('defs')
      .append('clipPath')
      .attr('id', 'plot-clip')
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', xMax)
      .attr('height', yMax);

    // Setup zoom
    const zoom = d3
      .zoom<Element, unknown>()
      .scaleExtent([0.5, 50])
      .translateExtent([
        [-xMax * 0.5, -yMax * 0.5],
        [xMax * 1.5, yMax * 1.5],
      ])
      .on('zoom', (event) => {
        setTransform(event.transform);
      });

    zoomRef.current = zoom;
    svg.call(zoom as any);

    // Axes and grid
    const xAxis = d3.axisBottom(xScale).ticks(8);
    const yAxis = d3.axisLeft(yScale).ticks(8);

    const xAxisGrid = d3.axisBottom(xScale).tickSize(-yMax).tickFormat(() => '').ticks(8);
    const yAxisGrid = d3.axisLeft(yScale).tickSize(-xMax).tickFormat(() => '').ticks(8);

    g.append('g')
      .attr('class', 'grid x-grid')
      .attr('transform', `translate(0,${yMax})`)
      .call(xAxisGrid)
      .selectAll('line')
      .attr('stroke', '#e5e5e5')
      .attr('stroke-dasharray', '2,2')
      .attr('opacity', 0.5);

    g.append('g')
      .attr('class', 'grid y-grid')
      .call(yAxisGrid)
      .selectAll('line')
      .attr('stroke', '#e5e5e5')
      .attr('stroke-dasharray', '2,2')
      .attr('opacity', 0.5);

    const xAxisG = g
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${yMax})`)
      .call(xAxis);

    xAxisG
      .append('text')
      .attr('x', xMax / 2)
      .attr('y', 45)
      .attr('fill', '#333')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text('Latent Dimension 1');

    const yAxisG = g.append('g').attr('class', 'y-axis').call(yAxis);

    yAxisG
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -yMax / 2)
      .attr('fill', '#333')
      .style('text-anchor', 'middle')
      .style('font-size', '13px')
      .style('font-weight', '600')
      .text('Latent Dimension 2');

    // Brush
    const brush = d3
      .brush<unknown>()
      .extent([
        [0, 0],
        [xMax, yMax],
      ])
      .on('end', (event) => {
        if (!event.selection) {
          setSelectedPoints(new Set());
          if (onBrush) onBrush([]);
          return;
        }

        const [[x0, y0], [x1, y1]] = event.selection;
        const selected = sampledData.filter((d) => {
          const x = xScale(d.x);
          const y = yScale(d.y);
          return x >= x0 && x <= x1 && y >= y0 && y <= y1;
        });

        setSelectedPoints(new Set(selected.map((d) => d.model_id)));
        if (onBrush) onBrush(selected);
      });

    g.append('g').attr('class', 'brush').call(brush);

    // Points with clip path
    const pointsGroup = g.append('g').attr('clip-path', 'url(#plot-clip)');

    // Performance: Simplified animations
    // Removed per-point delays and reduced duration for better performance
    const useAnimations = sampledData.length < 5000; // Disable animations for large datasets
    
    pointsGroup
      .selectAll<SVGCircleElement, ModelPoint>('circle')
      .data(sampledData, (d) => d.model_id)
      .join(
        (enter) => {
          const circles = enter
            .append('circle')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', useAnimations ? 0 : (d) => sizeScale(d))
            .attr('opacity', useAnimations ? 0 : 0.75);
          
          if (useAnimations) {
            circles
              .transition()
              .duration(300)
              .ease(d3.easeCubicOut)
              .attr('r', (d) => sizeScale(d))
              .attr('opacity', 0.75);
          }
          
          return circles;
        },
        (update) => {
          if (useAnimations) {
            return update.call((update) =>
              update
                .transition()
                .duration(200)
                .ease(d3.easeCubicOut)
                .attr('cx', (d) => xScale(d.x))
                .attr('cy', (d) => yScale(d.y))
                .attr('r', (d) => sizeScale(d))
            );
          } else {
            return update
              .attr('cx', (d) => xScale(d.x))
              .attr('cy', (d) => yScale(d.y))
              .attr('r', (d) => sizeScale(d));
          }
        },
        (exit) => {
          if (useAnimations) {
            return exit.call((exit) =>
              exit
                .transition()
                .duration(150)
                .ease(d3.easeCubicIn)
                .attr('r', 0)
                .attr('opacity', 0)
                .remove()
            );
          } else {
            return exit.remove();
          }
        }
      )
      .attr('fill', (d) => colorScale(d))
      .attr('stroke', (d) => {
        if (selectedPoints.has(d.model_id)) return '#000';
        if (hoveredPoint === d.model_id) return '#666';
        return 'rgba(255, 255, 255, 0.8)';
      })
      .attr('stroke-width', (d) => {
        if (selectedPoints.has(d.model_id)) return 2.5;
        if (hoveredPoint === d.model_id) return 2;
        return 0.8;
      })
      .attr('opacity', (d) => {
        if (selectedPoints.has(d.model_id)) return 1;
        if (hoveredPoint === d.model_id) return 0.95;
        return 0.75;
      })
      .style('cursor', 'pointer')
      .on('click', function (event, d) {
        event.stopPropagation();
        if (onPointClick) onPointClick(d);
      })
      .on('mouseover', function (event, d) {
        setHoveredPoint(d.model_id);
        d3.select(this)
          .transition()
          .duration(150)
          .attr('opacity', 1)
          .attr('stroke-width', 2.5)
          .attr('r', sizeScale(d) * 1.4);
        
        showTooltip(d, xScale(d.x), yScale(d.y));
      })
      .on('mouseout', function (event, d) {
        setHoveredPoint(null);
        if (!selectedPoints.has(d.model_id)) {
          d3.select(this)
            .transition()
            .duration(150)
            .attr('opacity', 0.75)
            .attr('stroke-width', 0.8)
            .attr('r', sizeScale(d));
        }
        hideTooltip();
      });

    // Update axes on zoom
    return () => {
      if (gRef.current) {
        const g = d3.select(gRef.current);
        g.select('.x-axis').call(xAxis as any);
        g.select('.y-axis').call(yAxis as any);
        g.select('.x-grid').call(xAxisGrid as any);
        g.select('.y-grid').call(yAxisGrid as any);
      }
    };
  }, [sampledData, width, height, margin, xScale, yScale, colorScale, sizeScale, selectedPoints, hoveredPoint, onPointClick, onBrush, resetZoom, showTooltip, hideTooltip]);

  return (
    <div className="scatter-plot-container">
      <svg 
        ref={svgRef} 
        width={width} 
        height={height} 
        className="scatter-plot-svg" 
      />
      <div 
        className="scatter-plot-controls-help"
        style={{ left: margin.left + 15 }}
      >
        <strong>Controls:</strong> Scroll to zoom • Drag to pan • Draw to select
      </div>
      {sampledData.length < data.length && (
        <div
          className="scatter-plot-sampling-notice"
          style={{ left: margin.left + 15 }}
        >
          Showing {sampledData.length.toLocaleString()} of {data.length.toLocaleString()} points
        </div>
      )}
      
      {/* Mini-map / Overview Map */}
      <MiniMap
        width={180}
        height={140}
        data={data}
        colorBy={colorBy}
        mainWidth={width}
        mainHeight={height}
        mainMargin={margin}
        transform={transform}
        onViewportChange={handleMiniMapViewportChange}
        xScaleBase={xScaleBase}
        yScaleBase={yScaleBase}
      />
    </div>
  );
}