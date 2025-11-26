/**
 * Zustand store for managing filter state across the application.
 * Centralizes filter logic for better performance and maintainability.
 */
import { create } from 'zustand';

export type ColorByOption = 'domain' | 'license' | 'family' | 'library' | 'library_name' | 'pipeline_tag' | 'cluster_id' | 'downloads' | 'likes' | 'family_depth' | 'trending_score' | 'licenses';
export type SizeByOption = 'downloads' | 'likes' | 'none';
export type ViewMode = '2d' | '3d' | 'scatter' | 'network' | 'distribution' | 'stacked' | 'heatmap';
export type RenderingStyle = 'embeddings' | 'sphere' | 'galaxy' | 'wave' | 'helix' | 'torus';
export type Theme = 'light' | 'dark';

interface FilterState {
  // Filters
  domains: string[];
  licenses: string[];
  dateRange: [number, number] | null;
  isBaseModel: boolean | null;
  minDownloads: number;
  minLikes: number;
  searchQuery: string;
  selectedClusters: number[];
  
  // View state
  colorBy: ColorByOption;
  sizeBy: SizeByOption;
  viewMode: ViewMode;
  colorScheme: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm';
  showLabels: boolean;
  zoomLevel: number;
  nodeDensity: number;
  renderingStyle: RenderingStyle;
  theme: Theme;
  
  // Actions
  setDomains: (domains: string[]) => void;
  setLicenses: (licenses: string[]) => void;
  setDateRange: (range: [number, number] | null) => void;
  setIsBaseModel: (value: boolean | null) => void;
  setMinDownloads: (value: number) => void;
  setMinLikes: (value: number) => void;
  setSearchQuery: (query: string) => void;
  setSelectedClusters: (clusters: number[]) => void;
  setColorBy: (value: ColorByOption) => void;
  setSizeBy: (value: SizeByOption) => void;
  setViewMode: (mode: ViewMode) => void;
  setColorScheme: (scheme: FilterState['colorScheme']) => void;
  setShowLabels: (show: boolean) => void;
  setZoomLevel: (level: number) => void;
  setNodeDensity: (density: number) => void;
  setRenderingStyle: (style: RenderingStyle) => void;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  resetFilters: () => void;
  
  // Computed
  getActiveFilterCount: () => number;
}

// Load theme from localStorage or default to light
const getInitialTheme = (): Theme => {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || saved === 'light') return saved;
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
  }
  return 'light';
};

export const useFilterStore = create<FilterState>((set, get) => ({
  // Initial state
  domains: [],
  licenses: [],
  dateRange: null,
  isBaseModel: null,
  minDownloads: 0,
  minLikes: 0,
  searchQuery: '',
  selectedClusters: [],
  colorBy: 'library_name',
  sizeBy: 'downloads',
  viewMode: '3d',
  colorScheme: 'viridis',
  showLabels: false,
  zoomLevel: 1,
  nodeDensity: 100,
  renderingStyle: 'embeddings',
  theme: getInitialTheme(),
  
  // Actions
  setDomains: (domains: string[]) => set({ domains }),
  setLicenses: (licenses: string[]) => set({ licenses }),
  setDateRange: (range: [number, number] | null) => set({ dateRange: range }),
  setIsBaseModel: (value: boolean | null) => set({ isBaseModel: value }),
  setMinDownloads: (value: number) => set({ minDownloads: value }),
  setMinLikes: (value: number) => set({ minLikes: value }),
  setSearchQuery: (query: string) => set({ searchQuery: query }),
  setSelectedClusters: (clusters: number[]) => set({ selectedClusters: clusters }),
  setColorBy: (value: ColorByOption) => set({ colorBy: value }),
  setSizeBy: (value: SizeByOption) => set({ sizeBy: value }),
  setViewMode: (mode: ViewMode) => set({ viewMode: mode }),
  setColorScheme: (scheme: FilterState['colorScheme']) => set({ colorScheme: scheme }),
  setShowLabels: (show: boolean) => set({ showLabels: show }),
  setZoomLevel: (level: number) => set({ zoomLevel: level }),
  setNodeDensity: (density: number) => set({ nodeDensity: density }),
  setRenderingStyle: (style: RenderingStyle) => set({ renderingStyle: style }),
  setTheme: (theme: Theme) => {
    set({ theme });
    if (typeof window !== 'undefined') {
      localStorage.setItem('theme', theme);
      document.documentElement.setAttribute('data-theme', theme);
    }
  },
  toggleTheme: () => {
    const currentTheme = get().theme;
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    get().setTheme(newTheme);
  },
  
  resetFilters: () => set({
    domains: [],
    licenses: [],
    dateRange: null,
    isBaseModel: null,
    minDownloads: 0,
    minLikes: 0,
    searchQuery: '',
    selectedClusters: [],
  }),
  
  getActiveFilterCount: () => {
    const state = get();
    let count = 0;
    if (state.domains.length > 0) count++;
    if (state.licenses.length > 0) count++;
    if (state.dateRange !== null) count++;
    if (state.isBaseModel !== null) count++;
    if (state.minDownloads > 0) count++;
    if (state.minLikes > 0) count++;
    if (state.searchQuery.length > 0) count++;
    if (state.selectedClusters.length > 0) count++;
    return count;
  },
}));

