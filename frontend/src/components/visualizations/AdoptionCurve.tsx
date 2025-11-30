import React, { useMemo, useCallback } from 'react';
import { Group } from '@visx/group';
import { AreaClosed, LinePath } from '@visx/shape';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { scaleTime, scaleLinear } from '@visx/scale';
import { useTooltip, TooltipWithBounds, defaultStyles } from '@visx/tooltip';
import { localPoint } from '@visx/event';
import { bisector } from 'd3-array';
import './AdoptionCurve.css';

export interface AdoptionDataPoint {
  date: Date;
  downloads: number;
  modelId: string;
}

interface ProcessedAdoptionDataPoint extends AdoptionDataPoint {
  cumulativeDownloads: number;
}

interface FamilyAdoptionData {
  family: string;
  data: AdoptionDataPoint[];
  color?: string;
}

interface AdoptionCurveProps {
  data: AdoptionDataPoint[];
  selectedModel?: string;
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  // Comparison mode: multiple families
  families?: FamilyAdoptionData[];
}

const defaultMargin = { top: 20, right: 20, bottom: 60, left: 80 };

const bisectDate = bisector<ProcessedAdoptionDataPoint, Date>((d) => d.date).left;

// Color palette for multiple families
const FAMILY_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

export default function AdoptionCurve({
  data,
  selectedModel,
  width = 800,
  height = 400,
  margin = defaultMargin,
  families,
}: AdoptionCurveProps) {
  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip,
  } = useTooltip<ProcessedAdoptionDataPoint>();

  // Process data: calculate cumulative downloads for single family mode
  const processedData: ProcessedAdoptionDataPoint[] = useMemo(() => {
    if (families && families.length > 0) return []; // Use families data instead
    
    if (!data || data.length === 0) return [];
    
    const sorted = [...data].sort((a, b) => a.date.getTime() - b.date.getTime());
    let cumulative = 0;
    
    return sorted.map((point) => {
      cumulative += point.downloads;
      return {
        ...point,
        cumulativeDownloads: cumulative,
      };
    });
  }, [data, families]);

  // Process multiple families for comparison mode
  const processedFamilies = useMemo(() => {
    if (!families || families.length === 0) return [];
    
    return families.map((family, idx) => {
      const sorted = [...family.data].sort((a, b) => a.date.getTime() - b.date.getTime());
      let cumulative = 0;
      
      const processed = sorted.map((point) => {
        cumulative += point.downloads;
        return {
          ...point,
          cumulativeDownloads: cumulative,
        };
      });
      
      return {
        family: family.family,
        data: processed,
        color: family.color || FAMILY_COLORS[idx % FAMILY_COLORS.length],
      };
    });
  }, [families]);

  // Calculate scales - handle both single and multi-family modes
  const allDates = useMemo(() => {
    if (processedFamilies.length > 0) {
      return processedFamilies.flatMap(f => f.data.map(d => d.date));
    }
    return processedData.map(d => d.date);
  }, [processedData, processedFamilies]);

  const allMaxDownloads = useMemo(() => {
    if (processedFamilies.length > 0) {
      return Math.max(...processedFamilies.flatMap(f => f.data.map(d => d.cumulativeDownloads)));
    }
    return Math.max(...processedData.map(d => d.cumulativeDownloads));
  }, [processedData, processedFamilies]);

  const xScale = useMemo(() => {
    if (allDates.length === 0) {
      return scaleTime({
        domain: [new Date(), new Date()],
        range: [margin.left, width - margin.right],
      });
    }
    
    return scaleTime({
      domain: [new Date(Math.min(...allDates.map(d => d.getTime()))), new Date(Math.max(...allDates.map(d => d.getTime())))],
      range: [margin.left, width - margin.right],
    });
  }, [allDates, width, margin]);

  const yScale = useMemo(() => {
    if (allDates.length === 0) {
      return scaleLinear({
        domain: [0, 1],
        range: [height - margin.bottom, margin.top],
      });
    }
    
    return scaleLinear({
      domain: [0, allMaxDownloads * 1.1],
      range: [height - margin.bottom, margin.top],
    });
  }, [allDates.length, allMaxDownloads, height, margin]);

  const isComparisonMode = processedFamilies.length > 0;

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGRectElement>) => {
      const coords = localPoint(event.currentTarget.ownerSVGElement!, event);
      if (!coords) return;

      const x0 = xScale.invert(coords.x - margin.left);
      const date = x0 instanceof Date ? x0 : new Date(x0);

      if (isComparisonMode) {
        // Find closest point across all families
        let closestPoint: ProcessedAdoptionDataPoint | null = null;
        let closestFamily: string | null = null;
        let minDistance = Infinity;

        processedFamilies.forEach((family) => {
          const index = bisectDate(family.data, date, 1);
          const a = family.data[index - 1];
          const b = family.data[index];
          
          let point: ProcessedAdoptionDataPoint | null = null;
          if (!b) {
            point = a;
          } else if (!a) {
            point = b;
          } else {
            point = date.getTime() - a.date.getTime() > b.date.getTime() - date.getTime() ? b : a;
          }

          if (point) {
            const distance = Math.abs(point.date.getTime() - date.getTime());
            if (distance < minDistance) {
              minDistance = distance;
              closestPoint = point;
              closestFamily = family.family;
            }
          }
        });

        if (closestPoint && closestFamily) {
          const tooltipDataWithFamily = Object.assign({}, closestPoint, { family: closestFamily });
          showTooltip({
            tooltipData: tooltipDataWithFamily as any,
            tooltipLeft: coords.x,
            tooltipTop: coords.y,
          });
        }
      } else {
        const index = bisectDate(processedData, date, 1);
        const a = processedData[index - 1];
        const b = processedData[index];
        
        let closestPoint: ProcessedAdoptionDataPoint | null = null;
        if (!b) {
          closestPoint = a;
        } else if (!a) {
          closestPoint = b;
        } else {
          closestPoint = date.getTime() - a.date.getTime() > b.date.getTime() - date.getTime() ? b : a;
        }

        if (closestPoint) {
          showTooltip({
            tooltipData: closestPoint,
            tooltipLeft: coords.x,
            tooltipTop: coords.y,
          });
        }
      }
    },
    [processedData, processedFamilies, isComparisonMode, xScale, margin, showTooltip]
  );

  const hasData = isComparisonMode || processedData.length > 0;

  if (!hasData) {
    return (
      <div className="adoption-curve-empty">
        <p>No adoption data available</p>
      </div>
    );
  }

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  return (
    <div className="adoption-curve-container">
      {/* Legend for comparison mode - positioned nicely */}
      {isComparisonMode && processedFamilies.length > 0 && (
        <div className="adoption-legend">
          <div className="legend-title">Families</div>
          {processedFamilies.map((family) => (
            <div key={family.family} className="legend-item">
              <div 
                className="legend-color-line" 
                style={{ backgroundColor: family.color }}
              />
              <span className="legend-label">{family.family}</span>
            </div>
          ))}
        </div>
      )}
      <svg width={width} height={height}>
        <Group>
          {isComparisonMode ? (
            // Multi-family comparison mode
            <>
              {processedFamilies.map((family, familyIdx) => {
                const color = family.color;
                const rgbaColor = color + '33'; // Add alpha for area
                
                return (
                  <React.Fragment key={family.family}>
                    {/* Area under curve */}
                    <AreaClosed<ProcessedAdoptionDataPoint>
                      data={family.data}
                      x={(d) => xScale(d.date) ?? 0}
                      y={(d) => yScale(d.cumulativeDownloads) ?? 0}
                      yScale={yScale}
                      fill={rgbaColor}
                      stroke="none"
                    />
                    {/* Line path */}
                    <LinePath<ProcessedAdoptionDataPoint>
                      data={family.data}
                      x={(d) => xScale(d.date) ?? 0}
                      y={(d) => yScale(d.cumulativeDownloads) ?? 0}
                      stroke={color}
                      strokeWidth={2.5}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </React.Fragment>
                );
              })}
            </>
          ) : (
            // Single family mode
            <>
              {/* Area under curve */}
              <AreaClosed<ProcessedAdoptionDataPoint>
                data={processedData}
                x={(d) => xScale(d.date) ?? 0}
                y={(d) => yScale(d.cumulativeDownloads) ?? 0}
                yScale={yScale}
                fill="rgba(59, 130, 246, 0.2)"
                stroke="none"
              />
              {/* Line path */}
              <LinePath<ProcessedAdoptionDataPoint>
                data={processedData}
                x={(d) => xScale(d.date) ?? 0}
                y={(d) => yScale(d.cumulativeDownloads) ?? 0}
                stroke="#3b82f6"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {/* Data points */}
              {processedData.map((point, idx) => {
                const x = xScale(point.date) ?? 0;
                const y = yScale(point.cumulativeDownloads) ?? 0;
                const isSelected = selectedModel === point.modelId;
                
                return (
                  <circle
                    key={`${point.modelId}-${idx}`}
                    cx={x}
                    cy={y}
                    r={isSelected ? 6 : 4}
                    fill={isSelected ? "#ef4444" : "#3b82f6"}
                    stroke={isSelected ? "#fff" : "none"}
                    strokeWidth={isSelected ? 2 : 0}
                    style={{ cursor: 'pointer' }}
                  />
                );
              })}
            </>
          )}

          {/* X-axis */}
          <AxisBottom
            top={height - margin.bottom}
            scale={xScale}
            numTicks={width > 520 ? 8 : 4}
            stroke="#666"
            tickStroke="#666"
            tickLabelProps={() => ({
              fill: '#666',
              fontSize: 10,
              textAnchor: 'middle',
              dy: -2,
            })}
            label="Date"
            labelProps={{
              fill: '#333',
              fontSize: 11,
              textAnchor: 'middle',
              dy: 40,
            }}
          />

          {/* Y-axis */}
          <AxisLeft
            left={margin.left}
            scale={yScale}
            numTicks={5}
            tickFormat={(value) => {
              const num = Number(value);
              if (num >= 1000000) {
                return `${(num / 1000000).toFixed(1)}M`;
              } else if (num >= 1000) {
                return `${(num / 1000).toFixed(0)}K`;
              }
              return num.toString();
            }}
            stroke="#666"
            tickStroke="#666"
            tickLabelProps={() => ({
              fill: '#666',
              fontSize: 10,
              textAnchor: 'end',
              dx: -4,
              dy: 3,
            })}
            label="Cumulative Downloads"
            labelProps={{
              fill: '#333',
              fontSize: 11,
              textAnchor: 'middle',
              transform: 'rotate(-90)',
              dy: -50,
            }}
          />

          {/* Invisible rect for mouse tracking */}
          <rect
            x={margin.left}
            y={margin.top}
            width={innerWidth}
            height={innerHeight}
            fill="transparent"
            onMouseMove={handleMouseMove}
            onMouseLeave={hideTooltip}
          />
        </Group>
      </svg>

      {/* Tooltip */}
      {tooltipOpen && tooltipData && (
        <TooltipWithBounds
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            ...defaultStyles,
            backgroundColor: 'var(--bg-elevated, #ffffff)',
            color: 'var(--text-primary, #1a1a1a)',
            border: '1px solid var(--border-medium, #d0d0d0)',
            padding: '0.75rem',
            borderRadius: '4px',
            fontSize: '0.875rem',
            pointerEvents: 'none',
            boxShadow: 'var(--shadow-lg, 0 2px 8px rgba(0, 0, 0, 0.12))',
          }}
        >
          <div className="adoption-tooltip">
            {(tooltipData as any).family && (
              <div className="tooltip-family" style={{ color: processedFamilies.find(f => f.family === (tooltipData as any).family)?.color }}>
                {(tooltipData as any).family}
              </div>
            )}
            <div className="tooltip-model-id">{tooltipData.modelId}</div>
            <div className="tooltip-date">
              {tooltipData.date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
              })}
            </div>
            <div className="tooltip-stats">
              <div>Downloads: {tooltipData.downloads.toLocaleString()}</div>
              <div>Cumulative: {tooltipData.cumulativeDownloads.toLocaleString()}</div>
            </div>
          </div>
        </TooltipWithBounds>
      )}

    </div>
  );
}
