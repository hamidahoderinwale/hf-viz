/**
 * API utilities for fetching graph/network data
 */
import { API_BASE } from '../../config/api';
import { GraphNode, GraphLink, EdgeType } from '../../components/visualizations/ForceDirectedGraph';

// Re-export types for convenience
export type { EdgeType, GraphNode, GraphLink };

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
 * Includes retry logic for rate limiting (429 errors)
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
  
  // Retry logic for rate limiting
  const maxRetries = 3;
  const baseDelay = 2000;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await fetch(url);
    
    // Handle 429 (Too Many Requests) with exponential backoff
    if (response.status === 429) {
      if (attempt === maxRetries - 1) {
        let errorMessage = 'Rate limit exceeded. Please wait a moment and try again.';
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
          const retryAfter = response.headers.get('Retry-After');
          if (retryAfter) {
            errorMessage += ` Please wait ${retryAfter} seconds.`;
          }
        } catch {
          // If response is not JSON, use default message
        }
        throw new Error(errorMessage);
      }
      
      const retryAfter = response.headers.get('Retry-After');
      const delay = retryAfter 
        ? parseInt(retryAfter) * 1000 
        : baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      
      console.warn(`Rate limit hit (429). Retrying in ${Math.round(delay / 1000)}s... (attempt ${attempt + 1}/${maxRetries})`);
      await new Promise(resolve => setTimeout(resolve, delay));
      continue;
    }
    
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
  
  // Should never reach here
  throw new Error('Failed to fetch network graph after retries');
}

/**
 * Fetch full derivative network graph for ALL models in the database
 * Includes retry logic for rate limiting (429 errors)
 * 
 * Use minDownloads and maxNodes to reduce network size for better performance.
 */
export async function fetchFullDerivativeNetwork(
  options: {
    edgeTypes?: EdgeType[];
    includeEdgeAttributes?: boolean;
    minDownloads?: number;
    maxNodes?: number;
    usePrecomputed?: boolean;
  } = {}
): Promise<NetworkGraphResponse> {
  // Default to false for performance with large graphs
  const { 
    edgeTypes, 
    includeEdgeAttributes = false,
    minDownloads = 0,
    maxNodes,
    usePrecomputed = true
  } = options;

  const params = new URLSearchParams();
  if (edgeTypes && edgeTypes.length > 0) {
    params.append('edge_types', edgeTypes.join(','));
  }
  if (includeEdgeAttributes !== undefined) {
    params.append('include_edge_attributes', includeEdgeAttributes.toString());
  }
  if (minDownloads > 0) {
    params.append('min_downloads', minDownloads.toString());
  }
  if (maxNodes !== undefined) {
    params.append('max_nodes', maxNodes.toString());
  }
  if (usePrecomputed !== undefined) {
    params.append('use_precomputed', usePrecomputed.toString());
  }

  const url = `${API_BASE}/api/network/full-derivatives${params.toString() ? '?' + params.toString() : ''}`;
  
  // Retry logic for rate limiting
  const maxRetries = 3;
  const baseDelay = 2000; // Start with 2 seconds
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    let response: Response;
    try {
      response = await fetch(url);
    } catch (error: any) {
      if (attempt === maxRetries - 1) {
        throw new Error(`Network error: ${error.message || 'Failed to connect to server'}`);
      }
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, baseDelay * (attempt + 1)));
      continue;
    }
    
    // Handle 429 (Too Many Requests) with exponential backoff
    if (response.status === 429) {
      if (attempt === maxRetries - 1) {
        let errorMessage = 'Rate limit exceeded. Please wait a moment and try again.';
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
          // Check for Retry-After header
          const retryAfter = response.headers.get('Retry-After');
          if (retryAfter) {
            errorMessage += ` Please wait ${retryAfter} seconds.`;
          }
        } catch {
          // If response is not JSON, use default message
        }
        throw new Error(errorMessage);
      }
      
      // Calculate delay: exponential backoff with jitter
      const retryAfter = response.headers.get('Retry-After');
      const delay = retryAfter 
        ? parseInt(retryAfter) * 1000 
        : baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      
      console.warn(`Rate limit hit (429). Retrying in ${Math.round(delay / 1000)}s... (attempt ${attempt + 1}/${maxRetries})`);
      await new Promise(resolve => setTimeout(resolve, delay));
      continue;
    }
    
    if (!response.ok) {
      let errorMessage = `Failed to fetch full derivative network: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch {
        // If response is not JSON, use status text
      }
      throw new Error(errorMessage);
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
  
  // Should never reach here, but TypeScript needs it
  throw new Error('Failed to fetch full derivative network after retries');
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
