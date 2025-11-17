/**
 * Enhanced D3.js scatter plot with zoom, pan, brush selection, and smooth animations.
 * Interactive latent space navigator with dynamic interactions.
 */
import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../types';

interface EnhancedScatterPlotProps {
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

export default function EnhancedScatterPlot({
  width,
  height,
  data,
  colorBy,
  sizeBy,
  margin = defaultMargin,
  onPointClick,
  onBrush,
}: EnhancedScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<Set<string>>(new Set());
  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const zoomRef = useRef<d3.ZoomBehavior<Element, unknown> | null>(null);

  // Sample data for very large datasets to improve performance and navigability
  const sampledData = useMemo(() => {
    // Reduced limit for better sparsity and navigability
    const renderLimit = 30000; // Reduced from 50K to 30K
    if (data.length <= renderLimit) return data;
    
    // Use step-based sampling for better distribution
    const step = Math.ceil(data.length / renderLimit);
    const sampled: typeof data = [];
    for (let i = 0; i < data.length; i += step) {
      sampled.push(data[i]);
    }
    return sampled;
  }, [data]);

  const { xScaleBase, yScaleBase, colorScale, sizeScale, useLogSize } = useMemo(() => {
    if (sampledData.length === 0) {
      return {
        xScaleBase: d3.scaleLinear(),
        yScaleBase: d3.scaleLinear(),
        colorScale: d3.scaleOrdinal(),
        sizeScale: d3.scaleLinear(),
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

    // Color scale
    const isCategorical = colorBy === 'library_name' || colorBy === 'pipeline_tag';
    let colorScale: d3.ScaleOrdinal<string, string> | d3.ScaleSequential<string, never> | ((d: ModelPoint) => string);
    
    if (isCategorical) {
      const categories = Array.from(new Set(sampledData.map((d) => {
        if (colorBy === 'library_name') return d.library_name || 'unknown';
        return d.pipeline_tag || 'unknown';
      })));
      colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(categories);
    } else {
      const values = sampledData.map((d) => {
        if (colorBy === 'downloads') return d.downloads;
        return d.likes;
      });
      const extent = d3.extent(values) as [number, number];
      // Use logarithmic scale for downloads/likes (heavily skewed distributions)
      if (colorBy === 'downloads' || colorBy === 'likes') {
        const logExtent: [number, number] = [
          Math.log10(extent[0] + 1),
          Math.log10(extent[1] + 1)
        ];
        const originalScale = d3.scaleSequential(d3.interpolateViridis).domain(logExtent);
        // Wrap to apply log transform
        colorScale = ((d: ModelPoint) => {
          const val = colorBy === 'downloads' ? d.downloads : d.likes;
          return originalScale(Math.log10(val + 1));
        });
      } else {
        colorScale = d3.scaleSequential(d3.interpolateViridis).domain(extent);
      }
    }

    // Size scale with logarithmic scaling for better representation of skewed distributions
    const sizeValues = sampledData.map((d) => {
      if (sizeBy === 'downloads') return d.downloads;
      if (sizeBy === 'likes') return d.likes;
      return 10;
    });
    const sizeExtent = d3.extent(sizeValues) as [number, number];
    // Use logarithmic scale for downloads/likes
    const useLogSize = sizeBy === 'downloads' || sizeBy === 'likes';
    let sizeScale: ReturnType<typeof d3.scaleSqrt> | ((d: ModelPoint) => number);
    if (useLogSize) {
      const logExtent: [number, number] = [
        Math.log10(sizeExtent[0] + 1),
        Math.log10(sizeExtent[1] + 1)
      ];
      const logScale = d3.scaleLinear().domain(logExtent).range([3, 20]);
      sizeScale = ((d: ModelPoint): number => {
        const val = sizeBy === 'downloads' ? d.downloads : d.likes;
        return logScale(Math.log10(val + 1));
      });
    } else {
      sizeScale = d3
        .scaleSqrt()
        .domain(sizeExtent)
        .range([3, 20]);
    }

    return { xScaleBase, yScaleBase, colorScale, sizeScale, useLogSize };
  }, [sampledData, width, height, margin, colorBy, sizeBy]);

  // Apply zoom transform to scales
  const xScale = useMemo(() => {
    const scale = xScaleBase.copy();
    return transform.rescaleX(scale);
  }, [xScaleBase, transform]);

  const yScale = useMemo(() => {
    const scale = yScaleBase.copy();
    return transform.rescaleY(scale);
  }, [yScaleBase, transform]);

  // Reset zoom handler
  const resetZoom = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(750).call(
        zoomRef.current.transform as any,
        d3.zoomIdentity
      );
      setTransform(d3.zoomIdentity);
    }
  }, []);

  useEffect(() => {
    if (!svgRef.current || sampledData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Add zoom controls
    const controls = svg
      .append('g')
      .attr('class', 'zoom-controls')
      .attr('transform', `translate(${width - 120}, 20)`);

    controls
      .append('rect')
      .attr('width', 100)
      .attr('height', 60)
      .attr('fill', 'rgba(255, 255, 255, 0.95)')
      .attr('stroke', '#d0d0d0')
      .attr('rx', 2)
      .style('cursor', 'pointer');

    controls
      .append('text')
      .attr('x', 50)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text('Zoom Controls');

    const zoomIn = controls
      .append('g')
      .attr('class', 'zoom-in')
      .attr('transform', 'translate(20, 30)')
      .style('cursor', 'pointer');

    zoomIn
      .append('rect')
      .attr('width', 25)
      .attr('height', 25)
      .attr('fill', '#4a4a4a')
      .attr('rx', 2)
      .on('click', () => {
        if (svgRef.current && zoomRef.current) {
          const svgNode = svgRef.current;
          const [x, y] = [width / 2, height / 2];
          d3.select(svgNode).transition().duration(300).call(
            zoomRef.current.scaleBy as any,
            1.5
          );
        }
      });

    zoomIn
      .append('text')
      .attr('x', 12.5)
      .attr('y', 17)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '16px')
      .text('+');

    const zoomOut = controls
      .append('g')
      .attr('class', 'zoom-out')
      .attr('transform', 'translate(55, 30)')
      .style('cursor', 'pointer');

    zoomOut
      .append('rect')
      .attr('width', 25)
      .attr('height', 25)
      .attr('fill', '#6a6a6a')
      .attr('rx', 2)
      .on('click', () => {
        if (svgRef.current && zoomRef.current) {
          const svgNode = svgRef.current;
          d3.select(svgNode).transition().duration(300).call(
            zoomRef.current.scaleBy as any,
            1 / 1.5
          );
        }
      });

    zoomOut
      .append('text')
      .attr('x', 12.5)
      .attr('y', 17)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '18px')
      .text('−');

    const resetBtn = controls
      .append('g')
      .attr('class', 'reset-zoom')
      .attr('transform', 'translate(37.5, 55)')
      .style('cursor', 'pointer');

    resetBtn
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#6a6a6a')
      .text('Reset')
      .on('click', resetZoom);

    const g = svg
      .append('g')
      .attr('class', 'main-group')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    gRef.current = g.node() as SVGGElement;

    const xMax = width - margin.left - margin.right;
    const yMax = height - margin.top - margin.bottom;

    // Setup zoom behavior
    const zoom = d3
      .zoom<Element, unknown>()
      .scaleExtent([0.1, 20])
      .translateExtent([
        [-margin.left, -margin.top],
        [width - margin.right, height - margin.bottom],
      ])
      .on('zoom', (event) => {
        setTransform(event.transform);
        g.attr('transform', `translate(${margin.left},${margin.top}) ${event.transform}`);
      });

    zoomRef.current = zoom;
    svg.call(zoom as any);

    // Add grid (will be updated on zoom)
    const xAxisGrid = d3
      .axisBottom(xScale)
      .tickSize(-yMax)
      .tickFormat(() => '');
    const yAxisGrid = d3
      .axisLeft(yScale)
      .tickSize(-xMax)
      .tickFormat(() => '');

    g.append('g')
      .attr('class', 'grid')
      .attr('data-axis', 'x')
      .attr('transform', `translate(0,${yMax})`)
      .call(xAxisGrid)
      .selectAll('line')
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.2)
      .attr('stroke', '#d0d0d0');

    g.append('g')
      .attr('class', 'grid')
      .attr('data-axis', 'y')
      .call(yAxisGrid)
      .selectAll('line')
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.2)
      .attr('stroke', '#d0d0d0');

    // Add axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    const xAxisG = g
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${yMax})`)
      .call(xAxis);

    xAxisG
      .append('text')
      .attr('x', xMax / 2)
      .attr('y', 40)
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Dimension 1');

    const yAxisG = g
      .append('g')
      .attr('class', 'y-axis')
      .call(yAxis);

    yAxisG
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -50)
      .attr('x', -yMax / 2)
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Dimension 2');

    // Add brush for selection (with zoom transform support)
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
        const selected = data.filter((d) => {
          const x = xScale(d.x);
          const y = yScale(d.y);
          return x >= x0 && x <= x1 && y >= y0 && y <= y1;
        });

        setSelectedPoints(new Set(selected.map((d) => d.model_id)));
        if (onBrush) onBrush(selected);
      });

    const brushG = g.append('g').attr('class', 'brush').call(brush);
    
    // Update brush on zoom
    svg.on('zoom.brush', () => {
      brushG.call(brush);
    });

    // Add points with smooth transitions
    const points = g
      .selectAll<SVGCircleElement, ModelPoint>('circle')
      .data(sampledData, (d) => d.model_id)
      .join(
        (enter) =>
          enter
            .append('circle')
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', 0)
            .attr('opacity', 0)
            .call((enter) =>
              enter
                .transition()
                .duration(500)
                .ease(d3.easeCubicOut)
                .attr('r', (d) => {
                  if (useLogSize) {
                    // Custom function that takes ModelPoint
                    return (sizeScale as (d: ModelPoint) => number)(d);
                  } else {
                    // D3 scale that takes a number
                    if (sizeBy === 'downloads') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(d.downloads);
                    if (sizeBy === 'likes') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(d.likes);
                    return 5;
                  }
                })
                .attr('opacity', (d) => {
                  if (selectedPoints.has(d.model_id)) return 1;
                  if (hoveredPoint === d.model_id) return 1;
                  return 0.7;
                })
            ),
        (update) =>
          update
            .transition()
            .duration(300)
            .ease(d3.easeCubicOut)
            .attr('cx', (d) => xScale(d.x))
            .attr('cy', (d) => yScale(d.y))
            .attr('r', (d) => {
              if (useLogSize) {
                // Custom function that takes ModelPoint
                return (sizeScale as (d: ModelPoint) => number)(d);
              } else {
                // D3 scale that takes a number
                if (sizeBy === 'downloads') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(d.downloads);
                if (sizeBy === 'likes') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(d.likes);
                return 5;
              }
            })
            .attr('opacity', (d) => {
              if (selectedPoints.has(d.model_id)) return 1;
              if (hoveredPoint === d.model_id) return 1;
              return 0.7;
            }),
        (exit) =>
          exit
            .transition()
            .duration(300)
            .ease(d3.easeCubicIn)
            .attr('r', 0)
            .attr('opacity', 0)
            .remove()
      )
      .attr('fill', (d) => {
        if (colorBy === 'library_name') {
          return (colorScale as d3.ScaleOrdinal<string, string>)(d.library_name || 'unknown');
        }
        if (colorBy === 'pipeline_tag') {
          return (colorScale as d3.ScaleOrdinal<string, string>)(d.pipeline_tag || 'unknown');
        }
        if (colorBy === 'downloads') {
          return (colorScale as d3.ScaleSequential<string, never>)(d.downloads);
        }
        return (colorScale as d3.ScaleSequential<string, never>)(d.likes);
      })
      .attr('stroke', (d) => {
        if (selectedPoints.has(d.model_id)) return '#1a1a1a';
        if (hoveredPoint === d.model_id) return '#4a4a4a';
        return '#ffffff';
      })
      .attr('stroke-width', (d) => {
        if (selectedPoints.has(d.model_id)) return 2.5;
        if (hoveredPoint === d.model_id) return 2;
        return 0.5;
      })
      .style('cursor', 'pointer')
      .style('pointer-events', 'all')
      .on('click', function (event, d) {
        event.stopPropagation();
        if (onPointClick) onPointClick(d);
      })
      .on('mouseover', function (event, d) {
        setHoveredPoint(d.model_id);
        const model = d as ModelPoint;
        d3.select(this)
          .transition()
          .duration(150)
          .attr('opacity', 1)
          .attr('stroke-width', 2)
          .attr('r', () => {
            let baseSize: number;
            if (useLogSize) {
              // Custom function that takes ModelPoint
              baseSize = (sizeScale as (d: ModelPoint) => number)(model);
            } else {
              // D3 scale that takes a number
              baseSize = sizeBy === 'downloads' 
                ? (sizeScale as ReturnType<typeof d3.scaleSqrt>)(model.downloads) 
                : sizeBy === 'likes' 
                ? (sizeScale as ReturnType<typeof d3.scaleSqrt>)(model.likes) 
                : 5;
            }
            return baseSize * 1.3;
          });
        
        // Show enhanced tooltip
        const [x, y] = [xScale(d.x), yScale(d.y)];
        const tooltip = g
          .append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${x + 15},${y - 15})`)
          .style('pointer-events', 'none');

        const tooltipBg = tooltip
          .append('rect')
          .attr('width', 220)
          .attr('height', 100)
          .attr('fill', 'rgba(26, 26, 26, 0.95)')
          .attr('rx', 2)
          .attr('opacity', 0)
          .transition()
          .duration(200)
          .attr('opacity', 1);

        tooltip
          .append('text')
          .attr('x', 12)
          .attr('y', 20)
          .attr('fill', 'white')
          .attr('font-size', '13px')
          .attr('font-weight', 'bold')
          .text(d.model_id.length > 30 ? d.model_id.substring(0, 30) + '...' : d.model_id);

        tooltip
          .append('text')
          .attr('x', 12)
          .attr('y', 40)
          .attr('fill', '#e0e0e0')
          .attr('font-size', '11px')
          .text(`Library: ${d.library_name || 'N/A'}`);

        tooltip
          .append('text')
          .attr('x', 12)
          .attr('y', 58)
          .attr('fill', '#e0e0e0')
          .attr('font-size', '11px')
          .text(`Pipeline: ${d.pipeline_tag || 'N/A'}`);

        tooltip
          .append('text')
          .attr('x', 12)
          .attr('y', 76)
          .attr('fill', '#d0d0d0')
          .attr('font-size', '11px')
          .text(`Downloads: ${d.downloads.toLocaleString()} | Likes: ${d.likes.toLocaleString()}`);

        tooltip
          .append('text')
          .attr('x', 12)
          .attr('y', 94)
          .attr('fill', '#c0c0c0')
          .attr('font-size', '10px')
          .text('Click for details');
      })
      .on('mouseout', function (event, d) {
        setHoveredPoint(null);
        if (!selectedPoints.has(d.model_id)) {
          const model = d as ModelPoint;
          d3.select(this)
            .transition()
            .duration(150)
            .attr('opacity', 0.7)
            .attr('stroke-width', 0.5)
            .attr('r', () => {
              if (useLogSize) {
                // Custom function that takes ModelPoint
                return (sizeScale as (d: ModelPoint) => number)(model);
              } else {
                // D3 scale that takes a number
                if (sizeBy === 'downloads') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(model.downloads);
                if (sizeBy === 'likes') return (sizeScale as ReturnType<typeof d3.scaleSqrt>)(model.likes);
                return 5;
              }
            });
        }
        g.selectAll('.tooltip').transition().duration(200).attr('opacity', 0).remove();
      });

    // Axes and grid will be updated automatically via the zoom transform
    // The scales (xScale, yScale) are already reactive to transform changes
  }, [
    data,
    xScale,
    yScale,
    colorScale,
    sizeScale,
    width,
    height,
    margin,
    colorBy,
    sizeBy,
    selectedPoints,
    hoveredPoint,
    transform,
    onPointClick,
    onBrush,
    resetZoom,
  ]);

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={svgRef} width={width} height={height} style={{ display: 'block' }} />
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: margin.left + 10,
          fontSize: '11px',
          color: '#6a6a6a',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '4px 8px',
          borderRadius: '2px',
          border: '1px solid #d0d0d0',
          fontFamily: "'Vend Sans', sans-serif",
        }}
      >
        <strong>Navigation:</strong> Scroll to zoom | Drag to pan | Click + drag to select
      </div>
    </div>
  );
}

