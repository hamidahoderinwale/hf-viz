/**
 * Client-side filtering engine for binary dataset.
 * Works with TypedArrays for fast vectorized operations.
 */

export interface ModelData {
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  domainId: Uint8Array;
  licenseId: Uint8Array;
  familyId: Uint16Array;
  flags: Uint8Array;
  modelIds: string[];
  domains: string[];
  licenses: string[];
}

export interface FilterCriteria {
  domains?: string[];
  licenses?: string[];
  isBaseModel?: boolean | null;
  families?: number[];
}

/**
 * Create a boolean mask based on filter criteria.
 * Returns Uint8Array where 1 = included, 0 = excluded.
 */
export function createFilterMask(
  data: ModelData,
  criteria: FilterCriteria
): Uint8Array {
  const numModels = data.x.length;
  const mask = new Uint8Array(numModels);

  // Initialize all to 1 (included)
  mask.fill(1);

  // Filter by domain
  if (criteria.domains && criteria.domains.length > 0) {
    const domainSet = new Set(criteria.domains);
    for (let i = 0; i < numModels; i++) {
      const domain = data.domains[data.domainId[i]];
      if (!domainSet.has(domain)) {
        mask[i] = 0;
      }
    }
  }

  // Filter by license
  if (criteria.licenses && criteria.licenses.length > 0) {
    const licenseSet = new Set(criteria.licenses);
    for (let i = 0; i < numModels; i++) {
      const license = data.licenses[data.licenseId[i]];
      if (!licenseSet.has(license)) {
        mask[i] = 0;
      }
    }
  }

  // Filter by base model flag
  if (criteria.isBaseModel !== undefined && criteria.isBaseModel !== null) {
    for (let i = 0; i < numModels; i++) {
      const isBase = (data.flags[i] & 0x01) !== 0;
      if (isBase !== criteria.isBaseModel) {
        mask[i] = 0;
      }
    }
  }

  // Filter by family
  if (criteria.families && criteria.families.length > 0) {
    const familySet = new Set(criteria.families);
    for (let i = 0; i < numModels; i++) {
      if (!familySet.has(data.familyId[i])) {
        mask[i] = 0;
      }
    }
  }

  return mask;
}

/**
 * Apply filter mask to data and return filtered arrays.
 */
export function applyMask(
  data: ModelData,
  mask: Uint8Array
): {
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  indices: number[];
  count: number;
} {
  const count = mask.reduce((sum, val) => sum + val, 0);
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const z = new Float32Array(count);
  const indices: number[] = [];

  let j = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) {
      x[j] = data.x[i];
      y[j] = data.y[i];
      z[j] = data.z[i];
      indices.push(i);
      j++;
    }
  }

  return { x, y, z, indices, count };
}

/**
 * Get model ID for a given index.
 */
export function getModelId(data: ModelData, index: number): string {
  return data.modelIds[index] || '';
}

/**
 * Get domain name for a given index.
 */
export function getDomain(data: ModelData, index: number): string {
  return data.domains[data.domainId[index]] || '';
}

/**
 * Get license name for a given index.
 */
export function getLicense(data: ModelData, index: number): string {
  return data.licenses[data.licenseId[index]] || '';
}

/**
 * Check if model is a base model (no parent).
 */
export function isBaseModel(data: ModelData, index: number): boolean {
  return (data.flags[index] & 0x01) !== 0;
}

/**
 * Check if model has children.
 */
export function hasChildren(data: ModelData, index: number): boolean {
  return (data.flags[index] & 0x04) !== 0;
}

