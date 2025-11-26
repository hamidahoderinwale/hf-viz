/**
 * Distribution view showing statistical distributions of model properties.
 * Uses D3.js for rendering (no external chart library needed).
 */
import React, { useMemo } from 'react';
import { ModelPoint } from '../../types';
import './DistributionView.css';

interface DistributionViewProps {
  data: ModelPoint[];
  width?: number;
  height?: number;
}

export default function DistributionView({ data, width = 800, height = 400 }: DistributionViewProps) {
  const distributions = useMemo(() => {
    if (data.length === 0) return null;

    // Library distribution
    const libraryDist = data.reduce((acc, model) => {
      const lib = model.library_name || 'Unknown';
      acc[lib] = (acc[lib] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Pipeline tag distribution
    const pipelineDist = data.reduce((acc, model) => {
      const pipeline = model.pipeline_tag || 'Unknown';
      acc[pipeline] = (acc[pipeline] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Downloads distribution (log scale buckets)
    const downloadsDist = data.reduce((acc, model) => {
      const downloads = model.downloads || 0;
      if (downloads === 0) {
        acc['0'] = (acc['0'] || 0) + 1;
      } else {
        const bucket = Math.floor(Math.log10(downloads));
        const label = `10^${bucket}`;
        acc[label] = (acc[label] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    // Likes distribution (log scale buckets)
    const likesDist = data.reduce((acc, model) => {
      const likes = model.likes || 0;
      if (likes === 0) {
        acc['0'] = (acc['0'] || 0) + 1;
      } else {
        const bucket = Math.floor(Math.log10(likes));
        const label = `10^${bucket}`;
        acc[label] = (acc[label] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    return {
      library: Object.entries(libraryDist)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, count]) => ({ name, count })),
      pipeline: Object.entries(pipelineDist)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, count]) => ({ name, count })),
      downloads: Object.entries(downloadsDist)
        .sort((a, b) => parseInt(a[0].replace('10^', '')) - parseInt(b[0].replace('10^', '')))
        .map(([name, count]) => ({ name, count })),
      likes: Object.entries(likesDist)
        .sort((a, b) => parseInt(a[0].replace('10^', '')) - parseInt(b[0].replace('10^', '')))
        .map(([name, count]) => ({ name, count })),
    };
  }, [data]);

  if (!distributions) {
    return (
      <div className="distribution-view">
        <div className="distribution-empty">No data to display</div>
      </div>
    );
  }

  const maxCount = Math.max(
    ...distributions.library.map(d => d.count),
    ...distributions.pipeline.map(d => d.count),
    ...distributions.downloads.map(d => d.count),
    ...distributions.likes.map(d => d.count)
  );

  const BarChart = ({ data, title }: { data: Array<{ name: string; count: number }>; title: string }) => (
    <div className="distribution-chart">
      <h4 className="distribution-chart-title">{title}</h4>
      <div className="distribution-bars">
        {data.map((item, idx) => (
          <div key={idx} className="distribution-bar-container">
            <div className="distribution-bar-label">{item.name}</div>
            <div className="distribution-bar-wrapper">
              <div
                className="distribution-bar"
                style={{ width: `${(item.count / maxCount) * 100}%` }}
              >
                <span className="distribution-bar-value">{item.count.toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="distribution-view">
      <div className="distribution-header">
        <h3>Model Distributions</h3>
        <div className="distribution-stats">
          Total Models: {data.length.toLocaleString()}
        </div>
      </div>
      <div className="distribution-grid">
        <BarChart data={distributions.library} title="Top Libraries" />
        <BarChart data={distributions.pipeline} title="Top Pipeline Tags" />
        <BarChart data={distributions.downloads} title="Downloads Distribution" />
        <BarChart data={distributions.likes} title="Likes Distribution" />
      </div>
    </div>
  );
}

