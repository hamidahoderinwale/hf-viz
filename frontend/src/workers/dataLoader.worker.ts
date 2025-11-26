/**
 * Web Worker for loading and parsing binary dataset format.
 * Runs off main thread to avoid blocking UI.
 */

interface BinaryHeader {
  magic: string;
  version: number;
  numModels: number;
  numDomains: number;
  numLicenses: number;
  numFamilies: number;
}

interface ModelData {
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

interface LoadMessage {
  type: 'load';
  binaryUrl: string;
  modelIdsUrl: string;
  metadataUrl: string;
}

interface FilterMessage {
  type: 'filter';
  data: ModelData;
  criteria: {
    domains?: string[];
    licenses?: string[];
    isBaseModel?: boolean | null;
  };
}

self.onmessage = async (e: MessageEvent<LoadMessage | FilterMessage>) => {
  const message = e.data;

  if (message.type === 'load') {
    try {
      // Fetch all files in parallel
      const [binaryResponse, modelIdsResponse, metadataResponse] = await Promise.all([
        fetch(message.binaryUrl),
        fetch(message.modelIdsUrl),
        fetch(message.metadataUrl),
      ]);

      if (!binaryResponse.ok || !modelIdsResponse.ok || !metadataResponse.ok) {
        throw new Error('Failed to fetch binary data files');
      }

      const arrayBuffer = await binaryResponse.arrayBuffer();
      const modelIds: string[] = await modelIdsResponse.json();
      const metadata: { domains: string[]; licenses: string[] } = await metadataResponse.json();

      const view = new DataView(arrayBuffer);
      let offset = 0;

      // Parse header (64 bytes)
      const magicBytes = new Uint8Array(arrayBuffer, offset, 5);
      // Convert Uint8Array to array for spread operator
      const magicArray = Array.from(magicBytes);
      const magic = String.fromCharCode(...magicArray);
      offset += 5;

      if (magic !== 'HFVIZ') {
        throw new Error(`Invalid binary format. Expected 'HFVIZ', got '${magic}'`);
      }

      const version = view.getUint8(offset);
      offset += 1;
      const numModels = view.getUint32(offset, true);
      offset += 4;
      const numDomains = view.getUint32(offset, true);
      offset += 4;
      const numLicenses = view.getUint32(offset, true);
      offset += 4;
      const numFamilies = view.getUint16(offset, true);
      offset += 2;
      offset += 50; // Skip reserved bytes

      // Parse domain lookup table (32 bytes per domain)
      const domains: string[] = [];
      for (let i = 0; i < numDomains; i++) {
        const domainBytes = new Uint8Array(arrayBuffer, offset, 32);
        const domain = new TextDecoder().decode(domainBytes).replace(/\0/g, '');
        domains.push(domain);
        offset += 32;
      }

      // Parse license lookup table (32 bytes per license)
      const licenses: string[] = [];
      for (let i = 0; i < numLicenses; i++) {
        const licenseBytes = new Uint8Array(arrayBuffer, offset, 32);
        const license = new TextDecoder().decode(licenseBytes).replace(/\0/g, '');
        licenses.push(license);
        offset += 32;
      }

      // Parse model records (16 bytes each: f32 x, f32 y, f32 z, u8 domain, u8 license, u16 family, u8 flags)
      const x = new Float32Array(numModels);
      const y = new Float32Array(numModels);
      const z = new Float32Array(numModels);
      const domainId = new Uint8Array(numModels);
      const licenseId = new Uint8Array(numModels);
      const familyId = new Uint16Array(numModels);
      const flags = new Uint8Array(numModels);

      for (let i = 0; i < numModels; i++) {
        x[i] = view.getFloat32(offset, true);
        offset += 4;
        y[i] = view.getFloat32(offset, true);
        offset += 4;
        z[i] = view.getFloat32(offset, true);
        offset += 4;
        domainId[i] = view.getUint8(offset);
        offset += 1;
        licenseId[i] = view.getUint8(offset);
        offset += 1;
        familyId[i] = view.getUint16(offset, true);
        offset += 2;
        flags[i] = view.getUint8(offset);
        offset += 1;
      }

      // Use metadata domains/licenses if available (more reliable)
      const finalDomains = metadata.domains && metadata.domains.length > 0 ? metadata.domains : domains;
      const finalLicenses = metadata.licenses && metadata.licenses.length > 0 ? metadata.licenses : licenses;

      const data: ModelData = {
        x,
        y,
        z,
        domainId,
        licenseId,
        familyId,
        flags,
        modelIds,
        domains: finalDomains,
        licenses: finalLicenses,
      };

      self.postMessage({ success: true, data, type: 'load' });
    } catch (error: any) {
      self.postMessage({
        success: false,
        error: error.message || 'Unknown error',
        type: 'load',
      });
    }
  } else if (message.type === 'filter') {
    // Filter data based on criteria
    const { data, criteria } = message;
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

    // Count filtered models
    const count = mask.reduce((sum, val) => sum + val, 0);

    // Extract filtered data
    const filteredX = new Float32Array(count);
    const filteredY = new Float32Array(count);
    const filteredZ = new Float32Array(count);
    const filteredIndices: number[] = [];

    let j = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 1) {
        filteredX[j] = data.x[i];
        filteredY[j] = data.y[i];
        filteredZ[j] = data.z[i];
        filteredIndices.push(i);
        j++;
      }
    }

    self.postMessage({
      success: true,
      type: 'filter',
      data: {
        x: filteredX,
        y: filteredY,
        z: filteredZ,
        indices: filteredIndices,
        count,
      },
    });
  }
};

