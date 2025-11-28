/**
 * Virtual scrolling component for large lists (search results, model lists).
 * Renders only visible items for 10-100x better performance.
 */
import React from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { SearchResult } from '../../types';

interface VirtualSearchResultsProps {
  results: SearchResult[];
  onSelect: (result: SearchResult) => void;
  selectedIndex?: number;
}

export const VirtualSearchResults: React.FC<VirtualSearchResultsProps> = ({
  results,
  onSelect,
  selectedIndex = -1,
}) => {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const result = results[index];
    const isSelected = index === selectedIndex;

    return (
      <div
        style={{
          ...style,
          cursor: 'pointer',
          padding: '8px 12px',
          backgroundColor: isSelected ? 'var(--accent-color)' : 'transparent',
          borderBottom: '1px solid var(--border-color)',
        }}
        onClick={() => onSelect(result)}
        onMouseEnter={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = 'var(--hover-color)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = 'transparent';
          }
        }}
      >
        <div style={{ fontWeight: 500 }}>{result.model_id}</div>
        {result.library_name && (
          <div style={{ fontSize: '0.875rem', opacity: 0.7, marginTop: '2px' }}>
            {result.library_name} {result.pipeline_tag && `• ${result.pipeline_tag}`}
          </div>
        )}
        {(result.downloads || result.likes) && (
          <div style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '2px' }}>
            {result.downloads?.toLocaleString()} downloads • {result.likes?.toLocaleString()} likes
          </div>
        )}
      </div>
    );
  };

  return (
    <AutoSizer>
      {({ height, width }: { height: number; width: number }) => (
        <List
          height={Math.min(height, 400)}
          itemCount={results.length}
          itemSize={70}
          width={width}
          overscanCount={5}
        >
          {Row}
        </List>
      )}
    </AutoSizer>
  );
};

export default VirtualSearchResults;


