/**
 * Virtual scrolling component for large lists (search results, model lists).
 * Renders only visible items for 10-100x better performance.
 */
import React from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { SearchResult } from '../../types';
import './VirtualSearchResults.css';

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
        style={style}
        className={`virtual-search-result ${isSelected ? 'selected' : ''}`}
        onClick={() => onSelect(result)}
      >
        <div className="virtual-search-result-title">{result.model_id}</div>
        {result.library_name && (
          <div className="virtual-search-result-meta">
            {result.library_name} {result.pipeline_tag && `• ${result.pipeline_tag}`}
          </div>
        )}
        {(result.downloads || result.likes) && (
          <div className="virtual-search-result-stats">
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


