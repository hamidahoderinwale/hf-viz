/**
 * File tree component for displaying model file structure.
 * Fetches and displays files from Hugging Face model repository.
 */
import React, { useState, useEffect } from 'react';
import './FileTree.css';

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

  useEffect(() => {
    const fetchFiles = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch file tree through our backend API (avoids CORS issues)
        // Use same API base as main app
        const apiBase = (window as any).__API_BASE__ || process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const response = await fetch(
          `${apiBase}/api/model/${encodeURIComponent(modelId)}/files?branch=main`
        );

        if (!response.ok) {
          throw new Error('File tree not available for this model');
        }

        const data = await response.json();
        
        // Convert flat list to tree structure
        const tree = buildFileTree(data);
        setFiles(tree);
      } catch (err: any) {
        setError(err instanceof Error ? err.message : 'Failed to load files');
        console.error('Error fetching file tree:', err);
      } finally {
        setLoading(false);
      }
    };

    if (modelId) {
      fetchFiles();
    }
  }, [modelId]);

  const buildFileTree = (fileList: any[]): FileNode[] => {
    const tree: FileNode[] = [];
    const pathMap = new Map<string, FileNode>();

    // Sort files by path
    const sortedFiles = [...fileList].sort((a, b) => a.path.localeCompare(b.path));

    for (const file of sortedFiles) {
      const parts = file.path.split('/');
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
            size: file.size,
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

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const getFileIcon = (node: FileNode): string => {
    if (node.type === 'directory') {
      return expandedPaths.has(node.path) ? '📂' : '📁';
    }
    const ext = node.path.split('.').pop()?.toLowerCase();
    const iconMap: Record<string, string> = {
      'py': '🐍',
      'json': '📄',
      'txt': '📝',
      'md': '📖',
      'yml': '⚙️',
      'yaml': '⚙️',
      'bin': '💾',
      'safetensors': '🔒',
      'pt': '🔥',
      'pth': '🔥',
      'onnx': '🧠',
      'pb': '🧠',
    };
    return iconMap[ext || ''] || '📄';
  };

  const renderNode = (node: FileNode, depth: number = 0): React.ReactNode => {
    const isExpanded = expandedPaths.has(node.path);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.path} className="file-tree-node" style={{ paddingLeft: `${depth * 1.5}rem` }}>
        <div
          className={`file-tree-item ${node.type} ${isExpanded ? 'expanded' : ''}`}
          onClick={() => node.type === 'directory' && toggleExpand(node.path)}
          style={{ cursor: node.type === 'directory' ? 'pointer' : 'default' }}
        >
          <span className="file-icon">{getFileIcon(node)}</span>
          <span className="file-name">{node.path.split('/').pop()}</span>
          {node.type === 'file' && node.size && (
            <span className="file-size">{formatFileSize(node.size)}</span>
          )}
          {node.type === 'directory' && (
            <span className="file-expand">{isExpanded ? '▼' : '▶'}</span>
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

  return (
    <div className="file-tree-container">
      <div className="file-tree-header">
        <strong>Repository Files</strong>
        <a
          href={`https://huggingface.co/${modelId}/tree/main`}
          target="_blank"
          rel="noopener noreferrer"
          className="file-tree-link"
        >
          View on Hugging Face →
        </a>
      </div>
      <div className="file-tree">
        {files.map((node) => renderNode(node))}
      </div>
    </div>
  );
}

