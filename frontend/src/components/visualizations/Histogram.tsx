/**
 * D3.js histogram for visualizing distributions of model attributes.
 */
import React, { useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { ModelPoint } from '../../types';

interface HistogramProps {
  width: number;
  height: number;
  data: ModelPoint[];
  attribute: 'downloads' | 'likes' | 'trending_score';
  margin?: { top: number; right: number; bottom: number; left: number };
}

const defaultMargin = { top: 20, right: 20, bottom: 40, left: 60 };

export default function Histogram({
  width,
  height,
  data,
  attribute,
  margin = defaultMargin,
}: HistogramProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const histogramData = useMemo(() => {
    const values = data
      .map((d) => {
        if (attribute === 'downloads') return d.downloads;
        if (attribute === 'likes') return d.likes;
        return d.trending_score || 0;
      })
      .filter((v) => v > 0);

    if (values.length === 0) return [];

    const max = d3.max(values) || 1;
    const bins = d3
      .bin()
      .domain([0, max])
      .thresholds(20)(values);

    return bins.map((bin) => ({
      x0: bin.x0 || 0,
      x1: bin.x1 || 0,
      length: bin.length,
    }));
  }, [data, attribute]);

  useEffect(() => {
    if (!svgRef.current || histogramData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const xMax = width - margin.left - margin.right;
    const yMax = height - margin.top - margin.bottom;

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([
        0,
        d3.max(histogramData, (d) => d.x1) || 1,
      ])
      .range([0, xMax]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(histogramData, (d) => d.length) || 1])
      .nice()
      .range([yMax, 0]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Bars
    g.selectAll('rect')
      .data(histogramData)
      .join('rect')
      .attr('x', (d) => xScale(d.x0))
      .attr('width', (d) => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
      .attr('y', (d) => yScale(d.length))
      .attr('height', (d) => yMax - yScale(d.length))
      .attr('fill', 'steelblue')
      .attr('opacity', 0.7)
      .on('mouseover', function (event, d) {
        d3.select(this).attr('opacity', 1);
      })
      .on('mouseout', function () {
        d3.select(this).attr('opacity', 0.7);
      });

    // X axis
    const xAxis = d3.axisBottom(xScale).tickFormat(d3.format('.2s'));
    g.append('g')
      .attr('transform', `translate(0,${yMax})`)
      .call(xAxis)
      .append('text')
      .attr('x', xMax / 2)
      .attr('y', 35)
      .attr('fill', 'currentColor')
      .style('text-anchor', 'middle')
      .text(attribute.replace('_', ' ').toUpperCase());

    // Y axis
    const yAxis = d3.axisLeft(yScale);
    g.append('g').call(yAxis).append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -50)
      .attr('x', -yMax / 2)
      .attr('fill', 'currentColor')
      .style('text-anchor', 'middle')
      .text('Count');
  }, [histogramData, width, height, margin, attribute]);

  return <svg ref={svgRef} width={width} height={height} />;
}

