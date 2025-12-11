/**
 * API utilities for fetching graph/network data
 */
import { API_BASE } from '../../config/api';
import { GraphNode, GraphLink, EdgeType } from '../../components/visualizations/ForceDirectedGraph';

export interface NetworkGraphResponse {
  nodes: GraphNode[];
  links: GraphLink[];
  statistics?: {
    nodes: number;
    edges: number;
    density: number;
    avg_degree: number;
    clustering: number;
  };
  root_model: string;
}

/**
 * Fetch family network graph for a specific model
 */
export async function fetchFamilyNetwork(
  modelId: string,
  options: {
    maxDepth?: number;
    edgeTypes?: EdgeType[];
    includeEdgeAttributes?: boolean;
  } = {}
): Promise<NetworkGraphResponse> {
  const { maxDepth, edgeTypes, includeEdgeAttributes = true } = options;

  const params = new URLSearchParams();
  if (maxDepth !== undefined) {
    params.append('max_depth', maxDepth.toString());
  }
  if (edgeTypes && edgeTypes.length > 0) {
    params.append('edge_types', edgeTypes.join(','));
  }
  if (includeEdgeAttributes !== undefined) {
    params.append('include_edge_attributes', includeEdgeAttributes.toString());
  }

  const url = `${API_BASE}/api/network/family/${encodeURIComponent(modelId)}${params.toString() ? '?' + params.toString() : ''}`;
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch network graph: ${response.statusText}`);
  }

  const data = await response.json();
  
  // Transform the response to match our types
  return {
    nodes: data.nodes || [],
    links: data.links || [],
    statistics: data.statistics,
    root_model: data.root_model || modelId,
  };
}

/**
 * Fetch full derivative network graph for ALL models in the database
 */
export async function fetchFullDerivativeNetwork(
  options: {
    edgeTypes?: EdgeType[];
    includeEdgeAttributes?: boolean;
  } = {}
): Promise<NetworkGraphResponse> {
  const { edgeTypes, includeEdgeAttributes = true } = options;

  const params = new URLSearchParams();
  if (edgeTypes && edgeTypes.length > 0) {
    params.append('edge_types', edgeTypes.join(','));
  }
  if (includeEdgeAttributes !== undefined) {
    params.append('include_edge_attributes', includeEdgeAttributes.toString());
  }

  const url = `${API_BASE}/api/network/full-derivatives${params.toString() ? '?' + params.toString() : ''}`;
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch full derivative network: ${response.statusText}`);
  }

  const data = await response.json();
  
  // Transform the response to match our types
  return {
    nodes: data.nodes || [],
    links: data.links || [],
    statistics: data.statistics,
    root_model: '', // No root model for full network
  };
}

/**
 * Get all available edge types from a graph response
 */
export function getAvailableEdgeTypes(links: GraphLink[]): Set<EdgeType> {
  const types = new Set<EdgeType>();
  links.forEach(link => {
    if (link.edge_types && link.edge_types.length > 0) {
      link.edge_types.forEach(type => types.add(type));
    } else if (link.edge_type) {
      types.add(link.edge_type);
    }
  });
  return types;
}
