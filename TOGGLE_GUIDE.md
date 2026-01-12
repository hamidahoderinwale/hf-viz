# Toggle Guide - What Each Toggle Does

## Navigation Tabs (Left Sidebar)

### 1. **Visualization** Tab (Default)
**What it does:**
- Shows the main 3D interactive visualization
- Displays models as points in a 3D embedding space
- Models closer together are more similar (based on embeddings)
- You can zoom, pan, rotate, and click on models

**What you see:**
- 3D scatter plot with thousands of models
- Color-coded by your selected attribute (library, pipeline, downloads, etc.)
- Size varies based on downloads/likes (if enabled)
- Mini-map in bottom-right corner

---

### 2. **Families** Tab
**What it does:**
- Switches to a different view focused on model families/organizations
- Shows adoption curves and family statistics
- Groups models by organization (e.g., "meta-llama", "google", "microsoft")

**What you see:**
- List of top model families by count
- Adoption curves showing how families grew over time
- Comparison mode to compare top 5 families
- Family depth distribution

**Key Features:**
- **Compare Top 5 Toggle**: Switches between single family view and comparison of top 5 families
- Click on a family to see its adoption curve
- Shows how models in each family were created over time

---

### 3. **Analytics** Tab
**What it does:**
- Shows statistics and rankings
- Displays top models by different metrics
- Shows trends and growth rates

**What you see:**
- Top models by downloads
- Top models by likes
- Trending models
- Newest models
- Largest families
- Fastest growing families (with growth rate %)

**Key Features:**
- Time range selector (24h, 7d, 30d) - filters data by time period
- Growth rate calculation shows % of models created in last 30 days

---

## Visualization Mode Toggle (Top Control Bar)

### **Embeddings** Mode (Default)
**What it does:**
- Shows models in semantic embedding space
- Uses UMAP coordinates (pre-computed 3D positions)
- Models positioned based on similarity (tags, descriptions, metadata)

**Controls available:**
- **Color By**: Change what determines point color
  - Family Depth
  - ML Library (transformers, pytorch, etc.)
  - Task Type (text-generation, image-classification, etc.)
  - Downloads (gradient)
  - Likes (gradient)
  
- **Size By**: Change what determines point size
  - Downloads
  - Likes
  - Uniform Size

- **Show All Models**: Toggle between sampled (150k) and all models

**What changes:**
- Point colors change based on selected attribute
- Point sizes change based on selected metric
- Visual grouping changes (e.g., libraries cluster together)

---

### **Force-directed Graph** Mode
**What it does:**
- Shows model relationships as a network graph
- Displays parent-child relationships (fine-tuning, quantization, etc.)
- Uses force-directed layout (models connected by edges)

**Controls available:**
- **Edge Type Filter**: Toggle which relationship types to show
  - Fine-tuned (blue) - models fine-tuned from a base model
  - Quantized (green) - quantized versions of models
  - Adapter (orange) - models with adapters added
  - Merged (purple) - merged models
  - Parent (gray) - generic parent relationships

- **Force Parameters** (Settings icon):
  - Link Distance: How far apart connected models are (50-200)
  - Charge Strength: Repulsion between models (-500 to -100)
  - Collision Radius: How much models avoid overlapping (0.5x to 2x)
  - Node Size: Size multiplier for nodes (0.5x to 2x)
  - Edge Opacity: Transparency of edges (0.1 to 1.0)

**What changes:**
- Graph layout changes based on force parameters
- Different relationship types appear/disappear based on edge type filter
- Node sizes and edge visibility adjust

---

## Color By Options (Embeddings Mode)

### **Family Depth**
- Colors models by their depth in the family tree
- Base models (depth 0) vs. fine-tuned models (depth 1, 2, 3...)
- Shows how models are related hierarchically

### **ML Library**
- Colors by library: transformers, pytorch, tensorflow, diffusers, etc.
- Each library gets a distinct color
- Shows which libraries are most popular

### **Task Type** (Pipeline Tag)
- Colors by task: text-generation, image-classification, etc.
- Groups models by what they're designed to do
- Shows task distribution in the ecosystem

### **Downloads** / **Likes**
- Uses a color gradient (viridis, plasma, inferno, cool-warm)
- Darker/lighter colors represent higher/lower values
- Shows popularity distribution

---

## Size By Options (Embeddings Mode)

### **By Downloads**
- Larger points = more downloads
- Logarithmic scaling (so differences are visible)
- Popular models stand out visually

### **By Likes**
- Larger points = more likes
- Shows community favorites
- Similar to downloads but reflects user engagement

### **Uniform Size**
- All points same size
- Useful when you want to focus on color patterns
- Better for seeing density patterns

---

## Edge Type Filter (Force-directed Graph Mode)

Each toggle button controls which relationship types are visible:

- **Fine-tuned** (Blue): Shows fine-tuning relationships
  - When OFF: Hides all fine-tuning edges
  - When ON: Shows fine-tuning connections

- **Quantized** (Green): Shows quantization relationships
  - When OFF: Hides quantized model connections
  - When ON: Shows quantization relationships

- **Adapter** (Orange): Shows adapter-based models
  - When OFF: Hides adapter relationships
  - When ON: Shows adapter connections

- **Merged** (Purple): Shows merged models
  - When OFF: Hides merge relationships
  - When ON: Shows merged model connections

- **Parent** (Gray): Shows generic parent relationships
  - When OFF: Hides parent connections
  - When ON: Shows parent-child relationships

**Effect:**
- Graph becomes simpler/hidden when types are disabled
- Helps focus on specific relationship types
- Reduces visual clutter

---

## Force Parameters (Force-directed Graph Mode)

### **Link Distance** (50-200)
- Controls how far apart connected nodes are
- Higher = more spread out graph
- Lower = more compact, clustered graph

### **Charge Strength** (-500 to -100)
- Controls repulsion between nodes
- More negative = stronger repulsion (nodes push apart)
- Less negative = weaker repulsion (nodes can cluster)

### **Collision Radius** (0.5x to 2x)
- Controls how much nodes avoid overlapping
- Higher = more spacing between nodes
- Lower = nodes can get closer together

### **Node Size** (0.5x to 2x)
- Multiplies the size of all nodes
- Useful for adjusting visibility
- Doesn't change graph layout, just appearance

### **Edge Opacity** (0.1 to 1.0)
- Controls transparency of edges
- Lower = more transparent (less visual clutter)
- Higher = more visible edges
- Useful for dense graphs

---

## Summary

**Navigation Tabs:**
- **Visualization**: Main 3D embedding space view
- **Families**: Family/organization analysis with adoption curves
- **Analytics**: Statistics, rankings, and trends

**Visualization Modes:**
- **Embeddings**: Semantic similarity space (default)
- **Force-directed Graph**: Relationship network view

**Key Toggles:**
- Color/Size controls change visual encoding
- Edge type filters show/hide relationship types
- Force parameters adjust graph layout
- Comparison mode (Families page) switches between single/comparison view

All toggles update the visualization in real-time without page reload!


