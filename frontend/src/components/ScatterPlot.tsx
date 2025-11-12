/**
 * Visx-based scatter plot component for model visualization.
 * Based on visx gallery examples: https://visx.airbnb.tech/gallery
 */
import React, { useMemo, useCallback } from 'react';
import { Group } from '@visx/group';
import { scaleLinear, scaleOrdinal } from '@visx/scale';
import { AxisBottom, AxisLeft } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import { Tooltip, useTooltip } from '@visx/tooltip';
import { LegendOrdinal } from '@visx/legend';
// Using circle elements directly instead of Point component
// Color schemes - using a predefined palette
const colorPalette = [
  '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
  '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
  '#ccebc5', '#ffed6f'
];
import { ModelPoint } from '../types';

interface ScatterPlotProps {
  width: number;
  height: number;
  data: ModelPoint[];
  colorBy: string;
  sizeBy: string;
  margin?: { top: number; right: number; bottom: number; left: number };
  onPointClick?: (model: ModelPoint) => void;
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
}: ScatterPlotProps) {
  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip,
  } = useTooltip<ModelPoint>();

  // Bounds
  const xMax = width - margin.left - margin.right;
  const yMax = height - margin.top - margin.bottom;

  // Scales
  const xScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [Math.min(...data.map(d => d.x)), Math.max(...data.map(d => d.x))],
        range: [0, xMax],
        nice: true,
      }),
    [data, xMax]
  );

  const yScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [Math.min(...data.map(d => d.y)), Math.max(...data.map(d => d.y))],
        range: [yMax, 0],
        nice: true,
      }),
    [data, yMax]
  );

  // Color scale
  const getColorValue = (d: ModelPoint) => {
    if (colorBy === 'library_name') return d.library_name || 'Unknown';
    if (colorBy === 'pipeline_tag') return d.pipeline_tag || 'Unknown';
    if (colorBy === 'downloads') return d.downloads;
    if (colorBy === 'likes') return d.likes;
    return 'All';
  };

  const colorValues = useMemo(() => data.map(getColorValue), [data, colorBy]);
  const isCategorical = colorBy === 'library_name' || colorBy === 'pipeline_tag';

  const colorScale = useMemo(() => {
    if (isCategorical) {
      const uniqueValues = Array.from(new Set(colorValues));
      return scaleOrdinal<string, string>({
        domain: uniqueValues,
        range: colorPalette,
      });
    } else {
      // For continuous, we'll use a linear scale with a color interpolator
      const min = Math.min(...(colorValues as number[]));
      const max = Math.max(...(colorValues as number[]));
      return scaleLinear<number, string>({
        domain: [min, max],
        range: ['#440154', '#fde725'], // Viridis-like colors
      });
    }
  }, [colorValues, isCategorical]);

  // Size scale
  const getSizeValue = (d: ModelPoint) => {
    if (sizeBy === 'downloads') return d.downloads;
    if (sizeBy === 'likes') return d.likes;
    if (sizeBy === 'trendingScore' && d.trending_score) return d.trending_score;
    return 10;
  };

  const sizeValues = useMemo(() => data.map(getSizeValue), [data, sizeBy]);
  const minSize = Math.min(...sizeValues);
  const maxSize = Math.max(...sizeValues);

  const sizeScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [minSize, maxSize],
        range: [5, 20],
      }),
    [minSize, maxSize]
  );

  // Handle point hover
  const handleMouseOver = useCallback(
    (event: React.MouseEvent, datum: ModelPoint) => {
      const coords = { x: event.clientX, y: event.clientY };
      showTooltip({
        tooltipLeft: coords.x,
        tooltipTop: coords.y,
        tooltipData: datum,
      });
    },
    [showTooltip]
  );

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height}>
        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows scale={yScale} width={xMax} strokeDasharray="3,3" stroke="#e0e0e0" />
          <GridColumns scale={xScale} height={yMax} strokeDasharray="3,3" stroke="#e0e0e0" />

          {/* Points */}
          {data.map((d, i) => {
            const x = xScale(d.x);
            const y = yScale(d.y);
            const color = isCategorical
              ? colorScale(getColorValue(d) as string)
              : colorScale(getColorValue(d) as number);
            const size = sizeScale(getSizeValue(d));

            return (
              <circle
                key={`point-${i}`}
                cx={x}
                cy={y}
                r={size / 2}
                fill={color}
                opacity={0.7}
                stroke="white"
                strokeWidth={0.5}
                onMouseOver={(e) => handleMouseOver(e, d)}
                onMouseOut={hideTooltip}
                onClick={() => onPointClick && onPointClick(d)}
                style={{ cursor: 'pointer' }}
              />
            );
          })}

          {/* Axes */}
          <AxisBottom
            top={yMax}
            scale={xScale}
            numTicks={5}
            label="Dimension 1"
            stroke="#333"
            tickStroke="#333"
          />
          <AxisLeft
            scale={yScale}
            numTicks={5}
            label="Dimension 2"
            stroke="#333"
            tickStroke="#333"
          />
        </Group>
      </svg>

      {/* Tooltip */}
      {tooltipOpen && tooltipData && (
        <Tooltip
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '12px',
          }}
        >
          <div>
            <strong>{tooltipData.model_id}</strong>
            <br />
            Library: {tooltipData.library_name || 'N/A'}
            <br />
            Pipeline: {tooltipData.pipeline_tag || 'N/A'}
            <br />
            Downloads: {tooltipData.downloads.toLocaleString()}
            <br />
            Likes: {tooltipData.likes.toLocaleString()}
          </div>
        </Tooltip>
      )}

      {/* Legend */}
      {isCategorical && (
        <div style={{ position: 'absolute', top: 10, right: 10 }}>
          <LegendOrdinal
            scale={colorScale as any}
            labelFormat={(label) => label}
            direction="column"
            style={{ fontSize: '12px' }}
          />
        </div>
      )}
    </div>
  );
}

