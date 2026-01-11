# Force-Directed Graph View - Current Status & Requirements

## Current State Analysis

### ✅ What EXISTS

1. **Force-Directed Graph View Implementation**
   - Located in: `frontend/src/App.tsx` (main visualization view)
   - Accessible via toggle button: "Embeddings" vs "Relationships"
   - Uses 3D force-directed graph components:
     - `ForceDirectedGraph3D.tsx` (for <10k nodes)
     - `ForceDirectedGraph3DInstanced.tsx` (for ≥10k nodes)
   - Also has 2D version: `ForceDirectedGraph.tsx` (not currently used in main view)

2. **Data Loading**
   - Fetches full derivative network via `fetchFullDerivativeNetwork()`
   - Automatically loads when `vizMode === 'force-graph'`
   - Shows loading states and error handling

3. **Edge Type Support**
   - Supports 5 edge types: `finetune`, `quantized`, `adapter`, `merge`, `parent`
   - Edge type filtering state exists (`enabledEdgeTypes`)
   - All edge types enabled by default

4. **Styling & Integration**
   - Uses same control bar layout as embeddings view
   - Shows graph statistics (node/edge counts) in control bar
   - Harmonious with dashboard style

### ❌ What's MISSING

1. **Edge Type Filtering Controls**
   - **Status**: Edge type filtering state exists but NO UI controls in main view
   - **Location**: Controls exist in `GraphPage.tsx` but not in `App.tsx` main view
   - **Need**: Add edge type toggle controls (checkboxes/buttons) in control bar when `vizMode === 'force-graph'`

2. **Configurable Force Parameters**
   - **Current**: Hardcoded in `ForceDirectedGraph.tsx`:
     - Link distance: 60-120 (based on edge type)
     - Charge strength: -300
     - Collision radius: 5 + sqrt(downloads)/200
   - **Need**: Add UI controls (sliders/inputs) for:
     - Link distance (base value)
     - Charge strength (repulsion)
     - Collision radius multiplier
     - Edge distance multipliers per type

3. **Default Display**
   - **Current**: Defaults to `'embeddings'` mode
   - **Line**: `const [vizMode, setVizMode] = useState<'embeddings' | 'force-graph'>('embeddings');`
   - **Question**: Should force-graph be the default? Or should it display by default in a specific context?

4. **2D vs 3D Option**
   - **Current**: Only shows 3D versions in main view
   - **Available**: 2D `ForceDirectedGraph.tsx` component exists but unused
   - **Reference**: The `force_directed_graph.html` reference uses 2D D3.js
   - **Need**: Add option to switch between 2D and 3D views

5. **Additional Parameters from Reference**
   - **Reference has**: Edge opacity controls, node size controls
   - **Current**: Node size based on downloads (hardcoded)
   - **Need**: Make node sizing configurable

## Comparison with Reference Implementation

### Reference (`force_directed_graph.html`):
- ✅ 2D D3.js force-directed layout
- ✅ Edge type filtering UI controls
- ✅ Configurable force parameters (link distance, charge strength)
- ✅ Edge opacity controls
- ✅ Node size controls
- ✅ Collapsible control panel

### Current Implementation:
- ✅ 3D Three.js force-directed layout (more advanced)
- ❌ No edge type filtering UI controls in main view
- ❌ Hardcoded force parameters
- ❌ No edge opacity controls
- ❌ Hardcoded node sizing
- ✅ Integrated into dashboard control bar

## Recommendations

### Priority 1: Essential Features
1. **Add Edge Type Filtering Controls**
   - Add edge type toggle buttons/checkboxes in control bar
   - Show when `vizMode === 'force-graph'`
   - Allow users to enable/disable specific edge types
   - Reuse pattern from `GraphPage.tsx` `EdgeTypeLegend` component

2. **Add 2D View Option**
   - Add toggle between 2D and 3D force-directed views
   - Use existing `ForceDirectedGraph.tsx` for 2D
   - Match reference implementation style

### Priority 2: Enhanced Configuration
3. **Make Force Parameters Configurable**
   - Add sliders for:
     - Base link distance (50-200)
     - Charge strength (-500 to -100)
     - Collision radius multiplier (0.5x to 2x)
   - Add per-edge-type distance multipliers

4. **Add Node Size Controls**
   - Add slider for node size scaling
   - Option to size by downloads, likes, or uniform

5. **Add Edge Opacity Controls**
   - Add slider for edge opacity (0.1 to 1.0)
   - Useful for dense graphs

### Priority 3: Default Behavior
6. **Consider Default Display**
   - Evaluate if force-graph should be default
   - Or add option to remember user preference
   - Or show force-graph by default for certain user types/contexts

## Implementation Plan

### Step 1: Add Edge Type Controls
- Create `EdgeTypeFilter` component (reuse from `GraphPage.tsx`)
- Add to control bar when `vizMode === 'force-graph'`
- Position after visualization mode toggle

### Step 2: Add 2D/3D Toggle
- Add toggle button in control bar
- Conditionally render `ForceDirectedGraph` (2D) vs `ForceDirectedGraph3D` (3D)
- Default to 2D to match reference, or add user preference

### Step 3: Add Force Parameter Controls
- Create `ForceParameterControls` component
- Add collapsible section in control bar
- Connect to force simulation parameters
- Update `ForceDirectedGraph.tsx` to accept configurable parameters

### Step 4: Add Node Size & Edge Opacity Controls
- Add sliders to control bar
- Update rendering components to use these values

## Files to Modify

1. `frontend/src/App.tsx`
   - Add edge type filter controls
   - Add 2D/3D toggle
   - Add force parameter controls
   - Add node size/opacity controls

2. `frontend/src/components/visualizations/ForceDirectedGraph.tsx`
   - Accept configurable force parameters as props
   - Accept node size multiplier
   - Accept edge opacity

3. `frontend/src/components/visualizations/ForceDirectedGraph3D.tsx`
   - Accept configurable force parameters as props
   - Accept node size multiplier
   - Accept edge opacity

4. `frontend/src/components/controls/` (new component)
   - Create `EdgeTypeFilter.tsx` (can reuse from `GraphPage.tsx`)
   - Create `ForceParameterControls.tsx`

## Current Code References

- Main view toggle: `App.tsx` lines 682-701
- Force graph rendering: `App.tsx` lines 883-920
- Edge type state: `App.tsx` line 102
- Force parameters (hardcoded): `ForceDirectedGraph.tsx` lines 148-179
- Edge type controls (reference): `GraphPage.tsx` lines 562-598

