/**
 * MessagePack decoder for binary API responses.
 * Provides 30-50% smaller payloads and faster parsing than JSON.
 */
import msgpack from 'msgpack-lite';
import { ModelPoint } from '../../types';

export async function fetchWithMsgPack<T = any>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options?.headers,
      'Accept': 'application/msgpack',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const contentType = response.headers.get('content-type');
  
  if (contentType?.includes('application/msgpack')) {
    const buffer = await response.arrayBuffer();
    return msgpack.decode(new Uint8Array(buffer)) as T;
  } else {
    // Fallback to JSON
    return response.json();
  }
}

export function decodeModelsMsgPack(data: Uint8Array): ModelPoint[] {
  const compact = msgpack.decode(data);
  
  if (!Array.isArray(compact)) {
    throw new Error('Invalid msgpack data: expected array');
  }

  return compact.map((cm: any) => ({
    model_id: cm.id,
    x: cm.x,
    y: cm.y,
    z: cm.z || 0,
    library_name: cm.lib,
    pipeline_tag: cm.pipe,
    downloads: cm.dl || 0,
    likes: cm.l || 0,
    trending_score: cm.ts,
    parent_model: cm.par,
    licenses: cm.lic,
    family_depth: cm.fd,
    cluster_id: cm.cid,
    tags: cm.tags,
    created_at: cm.ca,
  }));
}



