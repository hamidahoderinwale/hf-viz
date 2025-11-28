/**
 * File tree component for displaying model file structure.
 * Fetches and displays files from Hugging Face model repository.
 */
import React, { useState, useEffect, useMemo } from 'react';
import { getHuggingFaceFileTreeUrl } from '../../utils/api/hfUrl';
import './FileTree.css';

import { API_BASE } from '../../config/api';

interface FileNode {
  path: string;
  type: 'file' | 'directory';
  size?: number;
  children?: FileNode[];
}

interface FileTreeProps {
  modelId: string;
}

export default function FileTree({ modelId }: FileTreeProps) {
  const [files, setFiles] = useState<FileNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [fileTypeFilter, setFileTypeFilter] = useState<string>('all');
  const [showSearch, setShowSearch] = useState(false);
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!modelId) {
      setLoading(false);
      setError('No model ID provided');
      return;
    }

    const fetchFiles = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `${API_BASE}/api/model/${encodeURIComponent(modelId)}/files?branch=main`
        );

        if (response.status === 404) {
          throw new Error('File tree not available for this model');
        }
        
        if (response.status === 503) {
          throw new Error('Backend service unavailable');
        }

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Failed to load file tree: ${response.status} ${errorText}`);
        }

        const data = await response.json();
        
        if (!Array.isArray(data)) {
          throw new Error('Invalid response format');
        }
        
        // Convert flat list to tree structure
        const tree = buildFileTree(data);
        setFiles(tree);
      } catch (err: any) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load files';
        setError(errorMessage);
        // Only log in development
        if (process.env.NODE_ENV === 'development') {
          console.error('Error fetching file tree:', err);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchFiles();
  }, [modelId]);

  const buildFileTree = (fileList: any[]): FileNode[] => {
    if (!Array.isArray(fileList) || fileList.length === 0) {
      return [];
    }

    const tree: FileNode[] = [];
    const pathMap = new Map<string, FileNode>();

    // Sort files by path for consistent ordering
    const sortedFiles = [...fileList].sort((a, b) => {
      const pathA = a.path || '';
      const pathB = b.path || '';
      return pathA.localeCompare(pathB);
    });

    for (const file of sortedFiles) {
      if (!file.path) continue;

      const parts = file.path.split('/').filter((p: string) => p.length > 0);
      if (parts.length === 0) continue;

      let currentPath = '';
      let parent: FileNode | null = null;

      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        currentPath = currentPath ? `${currentPath}/${part}` : part;

        if (!pathMap.has(currentPath)) {
          const isDirectory = i < parts.length - 1;
          const node: FileNode = {
            path: currentPath,
            type: isDirectory ? 'directory' : 'file',
            size: isDirectory ? undefined : (file.size || undefined), // Only set size for files
            children: isDirectory ? [] : undefined,
          };

          pathMap.set(currentPath, node);

          if (parent) {
            parent.children!.push(node);
          } else {
            tree.push(node);
          }

          parent = node;
        } else {
          parent = pathMap.get(currentPath)!;
        }
      }
    }

    return tree;
  };

  const toggleExpand = (path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const expandAll = () => {
    const allPaths = new Set<string>();
    const collectPaths = (nodes: FileNode[]) => {
      nodes.forEach(node => {
        if (node.type === 'directory' && node.children) {
          allPaths.add(node.path);
          if (node.children.length > 0) {
            collectPaths(node.children);
          }
        }
      });
    };
    collectPaths(files);
    setExpandedPaths(allPaths);
  };

  const collapseAll = () => {
    setExpandedPaths(new Set());
  };

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  // Get all file extensions from the tree
  const getAllFileExtensions = useMemo(() => {
    const extensions = new Set<string>();
    const collectExtensions = (nodes: FileNode[]) => {
      nodes.forEach(node => {
        if (node.type === 'file') {
          const ext = node.path.split('.').pop()?.toLowerCase();
          if (ext) extensions.add(ext);
        }
        if (node.children) {
          collectExtensions(node.children);
        }
      });
    };
    collectExtensions(files);
    return Array.from(extensions).sort();
  }, [files]);

  // Auto-expand directories when searching
  useEffect(() => {
    if (searchQuery) {
      const pathsToExpand = new Set<string>();
      const findMatchingPaths = (nodes: FileNode[], query: string) => {
        nodes.forEach(node => {
          if (node.path.toLowerCase().includes(query.toLowerCase())) {
            // Expand all parent directories
            const parts = node.path.split('/');
            let currentPath = '';
            for (let i = 0; i < parts.length - 1; i++) {
              currentPath = currentPath ? `${currentPath}/${parts[i]}` : parts[i];
              pathsToExpand.add(currentPath);
            }
          }
          if (node.children) {
            findMatchingPaths(node.children, query);
          }
        });
      };
      findMatchingPaths(files, searchQuery);
      setExpandedPaths(pathsToExpand);
    }
  }, [searchQuery, files]);

  // Filter files based on search and file type
  const filterNodes = (nodes: FileNode[]): FileNode[] => {
    return nodes
      .map(node => {
        const matchesSearch = !searchQuery || 
          node.path.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesType = fileTypeFilter === 'all' || 
          (node.type === 'file' && node.path.toLowerCase().endsWith(`.${fileTypeFilter}`)) ||
          (node.type === 'directory');
        
        if (!matchesSearch || !matchesType) {
          return null;
        }

        const filteredChildren = node.children ? filterNodes(node.children) : undefined;
        const result: FileNode | null = filteredChildren && filteredChildren.length > 0
          ? { ...node, children: filteredChildren }
          : filteredChildren === undefined && matchesSearch && matchesType
          ? { ...node }
          : null;
        return result;
      })
      .filter((node): node is FileNode => node !== null);
  };

  const filteredFiles = useMemo(() => {
    if (!searchQuery && fileTypeFilter === 'all') return files;
    return filterNodes(files);
  }, [files, searchQuery, fileTypeFilter]);

  // Count total files
  const countFiles = (nodes: FileNode[]): number => {
    let count = 0;
    nodes.forEach(node => {
      if (node.type === 'file') count++;
      if (node.children) count += countFiles(node.children);
    });
    return count;
  };

  const totalFileCount = useMemo(() => countFiles(files), [files]);
  const visibleFileCount = useMemo(() => countFiles(filteredFiles), [filteredFiles]);

  // Keyboard shortcut for search (Cmd+K / Ctrl+K)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowSearch(true);
        setTimeout(() => searchInputRef.current?.focus(), 0);
      }
      if (e.key === 'Escape' && showSearch) {
        setShowSearch(false);
        setSearchQuery('');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showSearch]);

  const getFileIcon = (node: FileNode): string => {
    if (node.type === 'directory') {
      return expandedPaths.has(node.path) ? '▼' : '▶';
    }
    const ext = node.path.split('.').pop()?.toLowerCase();
    const iconMap: Record<string, string> = {
      'py': 'py',
      'json': 'json',
      'txt': 'txt',
      'md': 'md',
      'yml': 'yml',
      'yaml': 'yaml',
      'bin': 'bin',
      'safetensors': 'st',
      'pt': 'pt',
      'pth': 'pth',
      'onnx': 'onnx',
      'pb': 'pb',
    };
    return iconMap[ext || ''] || '•';
  };

  const copyFilePath = (path: string) => {
    navigator.clipboard.writeText(path).then(() => {
      // Show temporary feedback
      const button = document.querySelector(`[data-file-path="${path}"]`) as HTMLElement;
      if (button) {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        setTimeout(() => {
          if (button) button.textContent = originalText;
        }, 1000);
      }
    });
  };

  const getFileUrl = (path: string) => {
    return `https://huggingface.co/${modelId}/resolve/main/${path}`;
  };

  const renderNode = (node: FileNode, depth: number = 0): React.ReactNode => {
    const isExpanded = expandedPaths.has(node.path);
    const hasChildren = node.children && node.children.length > 0;
    const fileName = node.path.split('/').pop() || node.path;

    return (
      <div key={node.path} className="file-tree-node" style={{ paddingLeft: `${depth * 1.5}rem` }}>
        <div
          className={`file-tree-item ${node.type} ${isExpanded ? 'expanded' : ''}`}
          onClick={() => node.type === 'directory' && toggleExpand(node.path)}
          style={{ cursor: node.type === 'directory' ? 'pointer' : 'default' }}
        >
          <span className="file-icon">{getFileIcon(node)}</span>
          <span className="file-name" title={node.path}>{fileName}</span>
          {node.type === 'file' && node.size && (
            <span className="file-size">{formatFileSize(node.size)}</span>
          )}
          {node.type === 'directory' && (
            <span className="file-expand">{isExpanded ? '▼' : '▶'}</span>
          )}
          {node.type === 'file' && (
            <div className="file-actions" onClick={(e) => e.stopPropagation()}>
              <button
                className="file-action-btn"
                onClick={() => copyFilePath(node.path)}
                data-file-path={node.path}
                title="Copy file path"
                aria-label="Copy path"
              >
                Copy
              </button>
              <a
                href={getFileUrl(node.path)}
                target="_blank"
                rel="noopener noreferrer"
                className="file-action-btn"
                title="Download file"
                aria-label="Download"
                onClick={(e) => e.stopPropagation()}
              >
                Download
              </a>
            </div>
          )}
        </div>
        {isExpanded && hasChildren && (
          <div className="file-tree-children">
            {node.children!.map((child) => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="file-tree-container">
        <div className="file-tree-loading">Loading file tree...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="file-tree-container">
        <div className="file-tree-error">
          {error}
          <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#666' }}>
            File tree may not be available for this model.
          </div>
        </div>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="file-tree-container">
        <div className="file-tree-empty">No files found</div>
      </div>
    );
  }

  const hasDirectories = files.some(node => node.type === 'directory');

  return (
    <div className="file-tree-container">
      <div className="file-tree-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <strong>Repository Files</strong>
          <span className="file-count-badge">
            {visibleFileCount === totalFileCount 
              ? `${totalFileCount} file${totalFileCount !== 1 ? 's' : ''}`
              : `${visibleFileCount} of ${totalFileCount} files`}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            onClick={() => setShowSearch(!showSearch)}
            className="file-tree-button"
            title="Search files (Cmd+K)"
            aria-label="Search"
          >
            🔍 Search
          </button>
          {hasDirectories && (
            <>
              <button
                onClick={expandAll}
                className="file-tree-button"
                title="Expand all directories"
                aria-label="Expand all"
              >
                Expand All
              </button>
              <button
                onClick={collapseAll}
                className="file-tree-button"
                title="Collapse all directories"
                aria-label="Collapse all"
              >
                Collapse All
              </button>
            </>
          )}
          <a
            href={getHuggingFaceFileTreeUrl(modelId, 'main')}
            target="_blank"
            rel="noopener noreferrer"
            className="file-tree-link"
          >
            View on HF →
          </a>
        </div>
      </div>

      {/* Search and Filter Bar */}
      {(showSearch || searchQuery || fileTypeFilter !== 'all') && (
        <div className="file-tree-filters">
          <div className="file-tree-search">
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search files... (Cmd+K)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="file-tree-search-input"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="file-tree-clear"
                aria-label="Clear search"
              >
                ✕
              </button>
            )}
          </div>
          {getAllFileExtensions.length > 0 && (
            <select
              value={fileTypeFilter}
              onChange={(e) => setFileTypeFilter(e.target.value)}
              className="file-tree-type-filter"
            >
              <option value="all">All file types</option>
              {getAllFileExtensions.map(ext => (
                <option key={ext} value={ext}>.{ext}</option>
              ))}
            </select>
          )}
        </div>
      )}

      <div className="file-tree">
        {filteredFiles.length === 0 ? (
          <div className="file-tree-empty">
            {searchQuery || fileTypeFilter !== 'all' 
              ? 'No files match your filters'
              : 'No files found'}
          </div>
        ) : (
          filteredFiles.map((node) => renderNode(node))
        )}
      </div>
    </div>
  );
}

