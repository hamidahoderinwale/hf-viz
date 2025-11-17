/**
 * Color utility for generating varied, distinguishable color schemes.
 * Supports categorical and continuous color scales.
 */

// Extended color palettes for better variety
export const CATEGORICAL_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5',
  '#6b6ecf', '#b5cf6b', '#bd9e39', '#e7969c', '#7b4173',
  '#a55194', '#ce6dbd', '#de9ed6', '#636363', '#8ca252',
  '#b5a252', '#d6616b', '#e7ba52', '#ad494a', '#843c39',
  '#d6616b', '#e7969c', '#e7ba52', '#b5cf6b', '#8ca252',
  '#637939', '#bd9e39', '#d6616b', '#e7969c', '#e7ba52',
];

// Color schemes for different features
export const LIBRARY_COLORS: Record<string, string> = {
  'transformers': '#1f77b4',
  'diffusers': '#ff7f0e',
  'sentence-transformers': '#2ca02c',
  'timm': '#d62728',
  'speechbrain': '#9467bd',
  'fairseq': '#8c564b',
  'espnet': '#e377c2',
  'asteroid': '#7f7f7f',
  'keras': '#bcbd22',
  'sklearn': '#17becf',
  'unknown': '#cccccc',
};

export const PIPELINE_COLORS: Record<string, string> = {
  'text-classification': '#1f77b4',
  'token-classification': '#ff7f0e',
  'question-answering': '#2ca02c',
  'summarization': '#d62728',
  'translation': '#9467bd',
  'text-generation': '#8c564b',
  'fill-mask': '#e377c2',
  'zero-shot-classification': '#7f7f7f',
  'automatic-speech-recognition': '#bcbd22',
  'text-to-speech': '#17becf',
  'image-classification': '#aec7e8',
  'object-detection': '#ffbb78',
  'image-segmentation': '#98df8a',
  'image-to-text': '#ff9896',
  'text-to-image': '#c5b0d5',
  'unknown': '#cccccc',
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
  
  // Viridis-like color scale (blue to yellow)
  const viridis = (t: number) => {
    const r = Math.floor(68 + (253 - 68) * t);
    const g = Math.floor(1 + (231 - 1) * t);
    const b = Math.floor(84 + (37 - 84) * t);
    return `rgb(${r}, ${g}, ${b})`;
  };
  
  // Plasma color scale (purple to yellow)
  const plasma = (t: number) => {
    const r = Math.floor(13 + (240 - 13) * t);
    const g = Math.floor(8 + (249 - 8) * t);
    const b = Math.floor(135 + (33 - 135) * t);
    return `rgb(${r}, ${g}, ${b})`;
  };
  
  // Inferno color scale (black to yellow)
  const inferno = (t: number) => {
    const r = Math.floor(0 + (252 - 0) * t);
    const g = Math.floor(0 + (141 - 0) * t);
    const b = Math.floor(4 + (89 - 4) * t);
    return `rgb(${r}, ${g}, ${b})`;
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

