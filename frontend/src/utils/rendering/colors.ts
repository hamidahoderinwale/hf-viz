/**
 * Color utility for generating varied, distinguishable color schemes.
 * Supports categorical and continuous color scales.
 */

// Extended color palettes - HIGHLY VIBRANT for dark mode visibility
export const CATEGORICAL_COLORS = [
  '#60a5fa', '#fbbf24', '#34d399', '#f87171', '#a78bfa', // Bright versions
  '#f472b6', '#22d3ee', '#a3e635', '#fb923c', '#818cf8',
  '#2dd4bf', '#c084fc', '#fb7185', '#38bdf8', '#4ade80',
  '#facc15', '#3b82f6', '#a855f7', '#ec4899', '#06b6d4',
  '#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3',
  '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
  '#ee5a24', '#0abde3', '#10ac84', '#ff6b81', '#7bed9f',
  '#70a1ff', '#5352ed', '#ff4757', '#2ed573', '#ffa502',
];

// Color schemes for different features - EXTRA VIBRANT for dark mode
// Grouped by semantic meaning: NLP (blues), Vision (greens), Audio (purples), Generative (reds/oranges)
export const LIBRARY_COLORS: Record<string, string> = {
  // NLP / Text frameworks - Bright Blues and Cyans
  'transformers': '#60a5fa',      // Bright blue - most common
  'sentence-transformers': '#22d3ee', // Bright cyan
  'fairseq': '#38bdf8',           // Sky blue
  'spacy': '#2dd4bf',             // Teal
  
  // Vision frameworks - Bright Greens and Limes
  'timm': '#4ade80',              // Bright green
  'torchvision': '#a3e635',       // Bright lime
  'mmdet': '#bef264',             // Yellow-green
  
  // Diffusion / Generative - Bright Oranges and Reds
  'diffusers': '#fb923c',         // Bright orange
  'stable-baselines3': '#fdba74', // Light orange
  
  // Audio frameworks - Bright Purples and Pinks
  'speechbrain': '#c084fc',       // Bright purple
  'espnet': '#e879f9',            // Bright fuchsia
  'asteroid': '#f472b6',          // Bright pink
  
  // ML frameworks - Bright Warm colors
  'keras': '#fbbf24',             // Bright amber
  'sklearn': '#facc15',           // Bright yellow
  'pytorch': '#f87171',           // Bright red
  
  // Other
  'unknown': '#cbd5e1',           // Light slate
};

export const PIPELINE_COLORS: Record<string, string> = {
  // Text tasks - Bright Blues
  'text-classification': '#60a5fa',
  'token-classification': '#93c5fd',
  'question-answering': '#38bdf8',
  'fill-mask': '#22d3ee',
  'text-generation': '#2dd4bf',
  'summarization': '#5eead4',
  'translation': '#99f6e4',
  'zero-shot-classification': '#a5f3fc',
  
  // Vision tasks - Bright Greens
  'image-classification': '#4ade80',
  'object-detection': '#86efac',
  'image-segmentation': '#bbf7d0',
  'image-to-text': '#bef264',
  
  // Generative tasks - Bright Oranges/Reds
  'text-to-image': '#fb923c',
  'image-to-image': '#fdba74',
  
  // Audio tasks - Bright Purples
  'automatic-speech-recognition': '#c084fc',
  'text-to-speech': '#d8b4fe',
  'audio-classification': '#e879f9',
  
  // Other
  'unknown': '#cbd5e1',
};

// Depth-based color scale - Multi-hue gradient for maximum visibility
// Root models are bright cyan, deepest are bright magenta
export function getDepthColorScale(maxDepth: number, isDarkMode: boolean = true): (depth: number) => string {
  return (depth: number) => {
    // Normalize depth to 0-1 range
    const normalized = Math.max(0, Math.min(1, depth / Math.max(maxDepth, 1)));
    
    if (isDarkMode) {
      // Dark mode: Use a vibrant multi-hue gradient (cyan -> green -> yellow -> orange -> pink)
      // This provides maximum distinguishability between depth levels
      if (normalized < 0.25) {
        // Cyan to Green
        const t = normalized * 4;
        return `rgb(${Math.floor(34 + (74 - 34) * t)}, ${Math.floor(211 + (222 - 211) * t)}, ${Math.floor(238 + (128 - 238) * t)})`;
      } else if (normalized < 0.5) {
        // Green to Yellow
        const t = (normalized - 0.25) * 4;
        return `rgb(${Math.floor(74 + (250 - 74) * t)}, ${Math.floor(222 + (204 - 222) * t)}, ${Math.floor(128 + (21 - 128) * t)})`;
      } else if (normalized < 0.75) {
        // Yellow to Orange
        const t = (normalized - 0.5) * 4;
        return `rgb(${Math.floor(250 + (251 - 250) * t)}, ${Math.floor(204 + (146 - 204) * t)}, ${Math.floor(21 + (60 - 21) * t)})`;
      } else {
        // Orange to Pink/Magenta
        const t = (normalized - 0.75) * 4;
        return `rgb(${Math.floor(251 + (244 - 251) * t)}, ${Math.floor(146 + (114 - 146) * t)}, ${Math.floor(60 + (182 - 60) * t)})`;
      }
    } else {
      // Light mode: Darker, more saturated colors
      if (normalized < 0.5) {
        const t = normalized * 2;
        return `rgb(${Math.floor(30 + (100 - 30) * t)}, ${Math.floor(100 + (50 - 100) * t)}, ${Math.floor(200 + (150 - 200) * t)})`;
      } else {
        const t = (normalized - 0.5) * 2;
        return `rgb(${Math.floor(100 + (150 - 100) * t)}, ${Math.floor(50 + (30 - 50) * t)}, ${Math.floor(150 + (100 - 150) * t)})`;
      }
    }
  };
}

// Continuous color scales - EXTRA VIBRANT for dark mode visibility
export function getContinuousColorScale(
  min: number,
  max: number,
  scheme: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm' = 'viridis',
  useLogScale: boolean = false
): (value: number) => string {
  const range = max - min || 1;
  const logMin = useLogScale && min > 0 ? Math.log10(min + 1) : min;
  const logMax = useLogScale && max > 0 ? Math.log10(max + 1) : max;
  const logRange = logMax - logMin || 1;
  
  // Viridis - Bright cyan to bright yellow (enhanced for dark mode)
  const viridis = (t: number) => {
    if (t < 0.33) {
      const s = t * 3;
      return `rgb(${Math.floor(68 + (32 - 68) * s)}, ${Math.floor(170 + (200 - 170) * s)}, ${Math.floor(220 + (170 - 220) * s)})`;
    } else if (t < 0.66) {
      const s = (t - 0.33) * 3;
      return `rgb(${Math.floor(32 + (120 - 32) * s)}, ${Math.floor(200 + (220 - 200) * s)}, ${Math.floor(170 + (90 - 170) * s)})`;
    } else {
      const s = (t - 0.66) * 3;
      return `rgb(${Math.floor(120 + (253 - 120) * s)}, ${Math.floor(220 + (231 - 220) * s)}, ${Math.floor(90 + (37 - 90) * s)})`;
    }
  };
  
  // Plasma - Bright purple to bright yellow
  const plasma = (t: number) => {
    if (t < 0.33) {
      const s = t * 3;
      return `rgb(${Math.floor(100 + (180 - 100) * s)}, ${Math.floor(50 + (50 - 50) * s)}, ${Math.floor(200 + (220 - 200) * s)})`;
    } else if (t < 0.66) {
      const s = (t - 0.33) * 3;
      return `rgb(${Math.floor(180 + (240 - 180) * s)}, ${Math.floor(50 + (100 - 50) * s)}, ${Math.floor(220 + (150 - 220) * s)})`;
    } else {
      const s = (t - 0.66) * 3;
      return `rgb(${Math.floor(240 + (255 - 240) * s)}, ${Math.floor(100 + (220 - 100) * s)}, ${Math.floor(150 + (50 - 150) * s)})`;
    }
  };
  
  // Inferno - Dark red to bright yellow
  const inferno = (t: number) => {
    if (t < 0.33) {
      const s = t * 3;
      return `rgb(${Math.floor(60 + (150 - 60) * s)}, ${Math.floor(20 + (40 - 20) * s)}, ${Math.floor(80 + (100 - 80) * s)})`;
    } else if (t < 0.66) {
      const s = (t - 0.33) * 3;
      return `rgb(${Math.floor(150 + (230 - 150) * s)}, ${Math.floor(40 + (100 - 40) * s)}, ${Math.floor(100 + (50 - 100) * s)})`;
    } else {
      const s = (t - 0.66) * 3;
      return `rgb(${Math.floor(230 + (255 - 230) * s)}, ${Math.floor(100 + (200 - 100) * s)}, ${Math.floor(50 + (70 - 50) * s)})`;
    }
  };
  
  // Cool-warm - Bright cyan to bright red
  const coolwarm = (t: number) => {
    if (t < 0.5) {
      const s = t * 2;
      return `rgb(${Math.floor(80 + (200 - 80) * s)}, ${Math.floor(180 + (200 - 180) * s)}, ${Math.floor(255 + (220 - 255) * s)})`;
    } else {
      const s = (t - 0.5) * 2;
      return `rgb(${Math.floor(200 + (255 - 200) * s)}, ${Math.floor(200 + (100 - 200) * s)}, ${Math.floor(220 + (100 - 220) * s)})`;
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

