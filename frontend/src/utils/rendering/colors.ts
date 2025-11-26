/**
 * Color utility for generating varied, distinguishable color schemes.
 * Supports categorical and continuous color scales.
 */

// Extended color palettes for better variety - Enhanced vibrancy
export const CATEGORICAL_COLORS = [
  '#2563eb', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
  '#14b8a6', '#a855f7', '#f43f5e', '#0ea5e9', '#22c55e',
  '#eab308', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4',
  '#6b6ecf', '#b5cf6b', '#bd9e39', '#e7969c', '#7b4173',
  '#a55194', '#ce6dbd', '#de9ed6', '#636363', '#8ca252',
  '#b5a252', '#d6616b', '#e7ba52', '#ad494a', '#843c39',
  '#d6616b', '#e7969c', '#e7ba52', '#b5cf6b', '#8ca252',
  '#637939', '#bd9e39', '#d6616b', '#e7969c', '#e7ba52',
];

// Color schemes for different features - Enhanced vibrancy
export const LIBRARY_COLORS: Record<string, string> = {
  'transformers': '#2563eb',
  'diffusers': '#f59e0b',
  'sentence-transformers': '#10b981',
  'timm': '#ef4444',
  'speechbrain': '#8b5cf6',
  'fairseq': '#ec4899',
  'espnet': '#06b6d4',
  'asteroid': '#84cc16',
  'keras': '#f97316',
  'sklearn': '#6366f1',
  'unknown': '#9ca3af',
};

export const PIPELINE_COLORS: Record<string, string> = {
  'text-classification': '#2563eb',
  'token-classification': '#f59e0b',
  'question-answering': '#10b981',
  'summarization': '#ef4444',
  'translation': '#8b5cf6',
  'text-generation': '#ec4899',
  'fill-mask': '#06b6d4',
  'zero-shot-classification': '#84cc16',
  'automatic-speech-recognition': '#f97316',
  'text-to-speech': '#6366f1',
  'image-classification': '#14b8a6',
  'object-detection': '#a855f7',
  'image-segmentation': '#f43f5e',
  'image-to-text': '#0ea5e9',
  'text-to-image': '#22c55e',
  'unknown': '#9ca3af',
};

// Continuous color scales with optional logarithmic scaling
export function getContinuousColorScale(
  min: number,
  max: number,
  scheme: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm' = 'viridis',
  useLogScale: boolean = false
): (value: number) => string {
  // Use logarithmic scaling for heavily skewed distributions (like downloads/likes)
  // This provides better visual representation of the data distribution
  const range = max - min || 1;
  const logMin = useLogScale && min > 0 ? Math.log10(min + 1) : min;
  const logMax = useLogScale && max > 0 ? Math.log10(max + 1) : max;
  const logRange = logMax - logMin || 1;
  
  // Viridis-like color scale (blue to yellow) - Enhanced vibrancy
  const viridis = (t: number) => {
    // Apply gamma correction for more vibrant colors
    const gamma = 0.7;
    const tGamma = Math.pow(t, gamma);
    const r = Math.floor(68 + (253 - 68) * tGamma);
    const g = Math.floor(1 + (231 - 1) * tGamma);
    const b = Math.floor(84 + (37 - 84) * tGamma);
    // Increase saturation slightly
    return `rgb(${Math.min(255, r)}, ${Math.min(255, g)}, ${Math.min(255, b)})`;
  };
  
  // Plasma color scale (purple to yellow) - Enhanced vibrancy
  const plasma = (t: number) => {
    const gamma = 0.7;
    const tGamma = Math.pow(t, gamma);
    const r = Math.floor(13 + (240 - 13) * tGamma);
    const g = Math.floor(8 + (249 - 8) * tGamma);
    const b = Math.floor(135 + (33 - 135) * tGamma);
    return `rgb(${Math.min(255, r)}, ${Math.min(255, g)}, ${Math.min(255, b)})`;
  };
  
  // Inferno color scale (black to yellow) - Enhanced vibrancy
  const inferno = (t: number) => {
    const gamma = 0.6;
    const tGamma = Math.pow(t, gamma);
    const r = Math.floor(0 + (252 - 0) * tGamma);
    const g = Math.floor(0 + (141 - 0) * tGamma);
    const b = Math.floor(4 + (89 - 4) * tGamma);
    return `rgb(${Math.min(255, r)}, ${Math.min(255, g)}, ${Math.min(255, b)})`;
  };
  
  // Cool-warm color scale (blue to red)
  const coolwarm = (t: number) => {
    if (t < 0.5) {
      // Cool (blue)
      const s = t * 2;
      const r = Math.floor(59 * s);
      const g = Math.floor(76 * s);
      const b = Math.floor(192 + (255 - 192) * s);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Warm (red)
      const s = (t - 0.5) * 2;
      const r = Math.floor(180 + (255 - 180) * s);
      const g = Math.floor(4 + (180 - 4) * s);
      const b = Math.floor(38 * (1 - s));
      return `rgb(${r}, ${g}, ${b})`;
    }
  };
  
  const schemes = { viridis, plasma, inferno, magma: inferno, coolwarm };
  const colorFn = schemes[scheme];
  
  return (value: number) => {
    let normalized: number;
    if (useLogScale && value > 0) {
      const logValue = Math.log10(value + 1);
      normalized = Math.max(0, Math.min(1, (logValue - logMin) / logRange));
    } else {
      normalized = Math.max(0, Math.min(1, (value - min) / range));
    }
    return colorFn(normalized);
  };
}

// Generate color map for categorical data
export function getCategoricalColorMap(
  categories: string[],
  colorScheme: 'default' | 'library' | 'pipeline' = 'default'
): Map<string, string> {
  const colorMap = new Map<string, string>();
  const uniqueCategories = Array.from(new Set(categories)).sort();
  
  // Use predefined color schemes if available
  if (colorScheme === 'library') {
    uniqueCategories.forEach((cat) => {
      colorMap.set(cat, LIBRARY_COLORS[cat.toLowerCase()] || getColorForCategory(cat, uniqueCategories));
    });
  } else if (colorScheme === 'pipeline') {
    uniqueCategories.forEach((cat) => {
      colorMap.set(cat, PIPELINE_COLORS[cat.toLowerCase()] || getColorForCategory(cat, uniqueCategories));
    });
  } else {
    // Default: assign colors from extended palette
    uniqueCategories.forEach((cat, i) => {
      colorMap.set(cat, getColorForCategory(cat, uniqueCategories));
    });
  }
  
  return colorMap;
}

// Get color for a category using consistent hashing
function getColorForCategory(category: string, allCategories: string[]): string {
  // Use hash of category name for consistent color assignment
  let hash = 0;
  for (let i = 0; i < category.length; i++) {
    hash = ((hash << 5) - hash) + category.charCodeAt(i);
    hash = hash & hash; // Convert to 32-bit integer
  }
  const index = Math.abs(hash) % CATEGORICAL_COLORS.length;
  return CATEGORICAL_COLORS[index];
}

// Get color for a model based on colorBy attribute
export function getModelColor(
  model: { library_name?: string | null; pipeline_tag?: string | null; downloads?: number; likes?: number },
  colorBy: string,
  colorScale?: (d: any) => string
): string {
  if (colorScale) {
    return colorScale(model);
  }
  
  if (colorBy === 'library_name') {
    return LIBRARY_COLORS[model.library_name?.toLowerCase() || 'unknown'] || '#cccccc';
  }
  
  if (colorBy === 'pipeline_tag') {
    return PIPELINE_COLORS[model.pipeline_tag?.toLowerCase() || 'unknown'] || '#cccccc';
  }
  
  // Default fallback
  return '#808080';
}

