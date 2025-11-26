"""
Network analysis module inspired by Open Syllabus Project.
Builds co-occurrence networks for models based on shared contexts.
Supports multiple relationship types: finetune, quantized, adapter, merge.
"""
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
import ast
from datetime import datetime


def _parse_parent_list(value) -> List[str]:
    """
    Parse parent model list from string/eval format.
    Handles both string representations and actual lists.
    """
    if pd.isna(value) or value == '' or str(value) == 'nan':
        return []
    
    try:
        if isinstance(value, str):
            if value.startswith('[') or value.startswith('('):
                parsed = ast.literal_eval(value)
            else:
                parsed = [value]
        else:
            parsed = value
        
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p and str(p) != 'nan']
        elif parsed:
            return [str(parsed)]
        else:
            return []
    except (ValueError, SyntaxError):
        return []


def _get_all_parents(row: pd.Series) -> Dict[str, List[str]]:
    """
    Extract all parent types from a row.
    Returns dict mapping relationship type to list of parent IDs.
    """
    parents = {}
    
    parent_columns = {
        'parent_model': 'parent',
        'finetune_parent': 'finetune',
        'quantized_parent': 'quantized',
        'adapter_parent': 'adapter',
        'merge_parent': 'merge'
    }
    
    for col, rel_type in parent_columns.items():
        if col in row:
            parent_list = _parse_parent_list(row.get(col))
            if parent_list:
                parents[rel_type] = parent_list
    
    return parents


class ModelNetworkBuilder:
    """
    Build network graphs for models based on co-occurrence patterns.
    Similar to Open Syllabus approach of connecting texts that appear together.
    Supports multiple relationship types: finetune, quantized, adapter, merge.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with model dataframe.
        
        Args:
            df: DataFrame with model data including model_id, library_name, 
                pipeline_tag, tags, parent_model, finetune_parent, quantized_parent,
                adapter_parent, merge_parent, downloads, likes, createdAt
        """
        self.df = df.copy()
        if 'model_id' not in self.df.columns:
            raise ValueError("DataFrame must contain 'model_id' column")
        
        if self.df.index.name != 'model_id':
            if 'model_id' in self.df.columns:
                self.df.set_index('model_id', drop=False, inplace=True)
    
    def get_top_models_by_field(
        self, 
        library: Optional[str] = None,
        pipeline_tag: Optional[str] = None,
        min_downloads: int = 0,
        min_likes: int = 0,
        n: int = 100
    ) -> List[Tuple[str, int]]:
        """
        Extract top models by field/library (similar to Open Syllabus inst_text).
        
        Args:
            library: Filter by library name (e.g., 'transformers', 'diffusers')
            pipeline_tag: Filter by pipeline tag (e.g., 'text-classification')
            min_downloads: Minimum downloads threshold
            min_likes: Minimum likes threshold
            n: Number of top models to return
            
        Returns:
            List of (model_id, frequency) tuples, sorted by frequency
        """
        filtered = self.df.copy()
        
        # Apply filters
        if library:
            filtered = filtered[filtered.get('library_name', '') == library]
        if pipeline_tag:
            filtered = filtered[filtered.get('pipeline_tag', '') == pipeline_tag]
        if min_downloads > 0:
            filtered = filtered[filtered.get('downloads', 0) >= min_downloads]
        if min_likes > 0:
            filtered = filtered[filtered.get('likes', 0) >= min_likes]
        
        # Calculate frequency (using downloads as proxy for "teaching frequency")
        # In Open Syllabus, this is citation count; here we use downloads
        if 'downloads' in filtered.columns:
            filtered = filtered.nlargest(n, 'downloads', keep='first')
            results = [
                (str(model_id), int(row.get('downloads', 0)))
                for model_id, row in filtered.iterrows()
            ]
        else:
            # Fallback to likes or just count
            if 'likes' in filtered.columns:
                filtered = filtered.nlargest(n, 'likes', keep='first')
                results = [
                    (str(model_id), int(row.get('likes', 0)))
                    for model_id, row in filtered.iterrows()
                ]
            else:
                results = [(str(model_id), 1) for model_id in filtered.index[:n]]
        
        return results
    
    def build_cooccurrence_network(
        self,
        model_ids: List[str],
        cooccurrence_method: str = 'parent_family'
    ) -> nx.Graph:
        """
        Build network graph based on co-occurrence patterns.
        
        Similar to Open Syllabus: connect models that appear together in same context.
        
        Args:
            model_ids: List of model IDs to include in network
            cooccurrence_method: Method for determining co-occurrence
                - 'parent_family': Models with same parent (siblings)
                - 'library': Models in same library
                - 'pipeline': Models with same pipeline tag
                - 'tags': Models sharing common tags
                - 'combined': Combination of all methods
                
        Returns:
            NetworkX Graph with nodes and weighted edges
        """
        graph = nx.Graph()
        
        # Filter to only requested models
        available_models = [mid for mid in model_ids if mid in self.df.index]
        if not available_models:
            return graph
        
        model_df = self.df.loc[available_models]
        
        # Build co-occurrence edges
        edges = Counter()
        
        if cooccurrence_method in ['parent_family', 'combined']:
            # Connect models with same parent (siblings)
            parent_groups = model_df.groupby('parent_model')
            for parent, group in parent_groups:
                if pd.notna(parent) and len(group) > 1:
                    model_list = group.index.tolist()
                    for tid1, tid2 in combinations(sorted(model_list), 2):
                        edges[(tid1, tid2)] += 1
        
        if cooccurrence_method in ['library', 'combined']:
            # Connect models in same library
            library_groups = model_df.groupby('library_name')
            for library, group in library_groups:
                if pd.notna(library) and len(group) > 1:
                    model_list = group.index.tolist()
                    for tid1, tid2 in combinations(sorted(model_list), 2):
                        edges[(tid1, tid2)] += 1
        
        if cooccurrence_method in ['pipeline', 'combined']:
            # Connect models with same pipeline tag
            pipeline_groups = model_df.groupby('pipeline_tag')
            for pipeline, group in pipeline_groups:
                if pd.notna(pipeline) and len(group) > 1:
                    model_list = group.index.tolist()
                    for tid1, tid2 in combinations(sorted(model_list), 2):
                        edges[(tid1, tid2)] += 1
        
        if cooccurrence_method in ['tags', 'combined']:
            # Connect models sharing common tags
            for idx, row in model_df.iterrows():
                tags = str(row.get('tags', '')).lower().split(',')
                tags = [t.strip() for t in tags if t.strip()]
                if len(tags) > 1:
                    # Find other models with overlapping tags
                    for other_idx, other_row in model_df.iterrows():
                        if idx == other_idx:
                            continue
                        other_tags = str(other_row.get('tags', '')).lower().split(',')
                        other_tags = [t.strip() for t in other_tags if t.strip()]
                        overlap = set(tags) & set(other_tags)
                        if len(overlap) > 0:
                            key = tuple(sorted([str(idx), str(other_idx)]))
                            edges[key] += len(overlap)
        
        # Add edges to graph with weights
        for (tid1, tid2), count in edges.items():
            graph.add_edge(str(tid1), str(tid2), weight=count)
        
        # Add nodes with attributes
        for model_id, row in model_df.iterrows():
            model_id_str = str(model_id)
            if model_id_str not in graph:
                graph.add_node(model_id_str)
            
            # Add node attributes
            graph.nodes[model_id_str]['title'] = self._format_title(row.get('model_id', ''))
            graph.nodes[model_id_str]['author'] = self._extract_author(row.get('model_id', ''))
            graph.nodes[model_id_str]['freq'] = int(row.get('downloads', 0))
            graph.nodes[model_id_str]['likes'] = int(row.get('likes', 0))
            graph.nodes[model_id_str]['library'] = str(row.get('library_name', '')) if pd.notna(row.get('library_name')) else ''
            graph.nodes[model_id_str]['pipeline'] = str(row.get('pipeline_tag', '')) if pd.notna(row.get('pipeline_tag')) else ''
        
        return graph
    
    def _format_title(self, model_id: str) -> str:
        """Format model ID as title (similar to Open Syllabus title_format)."""
        if not model_id:
            return ''
        # Extract just the model name part
        parts = str(model_id).split('/')
        title = parts[-1] if len(parts) > 1 else model_id
        # Clean up
        title = title.replace("/", "").replace(";", "")
        return title
    
    def _extract_author(self, model_id: str) -> str:
        """Extract author/organization from model ID (similar to Open Syllabus last_name)."""
        if not model_id:
            return ''
        parts = str(model_id).split('/')
        if len(parts) > 1:
            return parts[0]  # Organization/author is first part
        return ''
    
    def build_family_tree_network(
        self,
        root_model_id: str,
        max_depth: Optional[int] = 5,
        include_edge_attributes: bool = True,
        filter_edge_types: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Build directed graph of model family tree with multiple relationship types.
        
        Args:
            root_model_id: Root model to start from
            max_depth: Maximum depth to traverse. If None, traverses entire tree without limit.
            include_edge_attributes: Whether to calculate edge attributes (change in likes, downloads, etc.)
            filter_edge_types: List of edge types to include (e.g., ['finetune', 'quantized']). 
                              If None, includes all types.
            
        Returns:
            NetworkX DiGraph representing family tree with edge types and attributes
        """
        graph = nx.DiGraph()
        visited = set()
        
        children_index: Dict[str, List[Tuple[str, str]]] = {}
        for idx, row in self.df.iterrows():
            model_id = str(row.get('model_id', idx))
            all_parents = _get_all_parents(row)
            
            for rel_type, parent_list in all_parents.items():
                for parent_id in parent_list:
                    if parent_id not in children_index:
                        children_index[parent_id] = []
                    children_index[parent_id].append((model_id, rel_type))
        
        def add_family(current_id: str, depth: Optional[int]):
            if current_id in visited:
                return
            if depth is not None and depth <= 0:
                return
            visited.add(current_id)
            
            if current_id not in self.df.index:
                return
            
            row = self.df.loc[current_id]
            
            graph.add_node(str(current_id))
            graph.nodes[str(current_id)]['title'] = self._format_title(current_id)
            graph.nodes[str(current_id)]['freq'] = int(row.get('downloads', 0))
            graph.nodes[str(current_id)]['likes'] = int(row.get('likes', 0))
            graph.nodes[str(current_id)]['downloads'] = int(row.get('downloads', 0))
            graph.nodes[str(current_id)]['library'] = str(row.get('library_name', '')) if pd.notna(row.get('library_name')) else ''
            graph.nodes[str(current_id)]['pipeline'] = str(row.get('pipeline_tag', '')) if pd.notna(row.get('pipeline_tag')) else ''
            
            createdAt = row.get('createdAt')
            if pd.notna(createdAt):
                graph.nodes[str(current_id)]['createdAt'] = str(createdAt)
            
            all_parents = _get_all_parents(row)
            for rel_type, parent_list in all_parents.items():
                if filter_edge_types and rel_type not in filter_edge_types:
                    continue
                
                for parent_id in parent_list:
                    if parent_id in self.df.index:
                        graph.add_edge(parent_id, str(current_id))
                        graph[parent_id][str(current_id)]['edge_types'] = [rel_type]
                        graph[parent_id][str(current_id)]['edge_type'] = rel_type
                        
                        next_depth = depth - 1 if depth is not None else None
                        add_family(parent_id, next_depth)
            
            children = children_index.get(current_id, [])
            for child_id, rel_type in children:
                if filter_edge_types and rel_type not in filter_edge_types:
                    continue
                
                if str(child_id) not in visited:
                    if not graph.has_edge(str(current_id), child_id):
                        graph.add_edge(str(current_id), child_id)
                        graph[str(current_id)][child_id]['edge_types'] = [rel_type]
                        graph[str(current_id)][child_id]['edge_type'] = rel_type
                    else:
                        if rel_type not in graph[str(current_id)][child_id].get('edge_types', []):
                            graph[str(current_id)][child_id]['edge_types'].append(rel_type)
                    
                    next_depth = depth - 1 if depth is not None else None
                    add_family(child_id, next_depth)
        
        add_family(root_model_id, max_depth)
        
        if include_edge_attributes:
            self._add_edge_attributes(graph)
        
        return graph
    
    def _add_edge_attributes(self, graph: nx.DiGraph):
        """
        Add edge attributes like change in likes, downloads, time difference.
        Similar to the notebook's edge attribute calculation.
        """
        for edge in graph.edges():
            parent_model = edge[0]
            model_id = edge[1]
            
            if parent_model not in graph.nodes() or model_id not in graph.nodes():
                continue
            
            parent_likes = graph.nodes[parent_model].get('likes', 0)
            model_likes = graph.nodes[model_id].get('likes', 0)
            parent_downloads = graph.nodes[parent_model].get('downloads', 0)
            model_downloads = graph.nodes[model_id].get('downloads', 0)
            
            graph.edges[edge]['change_in_likes'] = model_likes - parent_likes
            if parent_likes != 0:
                graph.edges[edge]['percentage_change_in_likes'] = (model_likes - parent_likes) / parent_likes
            else:
                graph.edges[edge]['percentage_change_in_likes'] = np.nan
            
            graph.edges[edge]['change_in_downloads'] = model_downloads - parent_downloads
            if parent_downloads != 0:
                graph.edges[edge]['percentage_change_in_downloads'] = (model_downloads - parent_downloads) / parent_downloads
            else:
                graph.edges[edge]['percentage_change_in_downloads'] = np.nan
            
            parent_created = graph.nodes[parent_model].get('createdAt')
            model_created = graph.nodes[model_id].get('createdAt')
            
            if parent_created and model_created:
                try:
                    parent_dt = datetime.strptime(str(parent_created), '%Y-%m-%dT%H:%M:%S.%fZ')
                    model_dt = datetime.strptime(str(model_created), '%Y-%m-%dT%H:%M:%S.%fZ')
                    graph.edges[edge]['change_in_createdAt_days'] = (model_dt - parent_dt).days
                except (ValueError, TypeError):
                    graph.edges[edge]['change_in_createdAt_days'] = np.nan
            else:
                graph.edges[edge]['change_in_createdAt_days'] = np.nan
    
    def export_graphml(self, graph: nx.Graph, filename: str):
        """Export graph to GraphML format (like Open Syllabus)."""
        nx.write_graphml(graph, filename)
    
    def get_network_statistics(self, graph: nx.Graph) -> Dict:
        """Get network statistics."""
        if len(graph) == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0,
                'avg_degree': 0,
                'clustering': 0
            }
        
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            'clustering': nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0
        }
    
    def build_full_network(self, cooccurrence_method: str = 'combined') -> nx.Graph:
        """
        Build a full co-occurrence network for all models in the dataset.
        This creates a persistent graph that can be used for graph-based search.
        
        Args:
            cooccurrence_method: Method for determining co-occurrence
            
        Returns:
            NetworkX Graph with all models and their relationships
        """
        all_model_ids = self.df.index.tolist()
        return self.build_cooccurrence_network(all_model_ids, cooccurrence_method)
    
    def find_neighbors(
        self,
        model_id: str,
        graph: Optional[nx.Graph] = None,
        max_neighbors: int = 50,
        min_weight: float = 0.0
    ) -> List[Dict]:
        """
        Find neighbors of a model in the network (graph-based search).
        Similar to graph database queries for finding connected nodes.
        
        Args:
            model_id: Model to find neighbors for
            graph: Pre-built network graph (if None, builds one)
            max_neighbors: Maximum number of neighbors to return
            min_weight: Minimum edge weight threshold
            
        Returns:
            List of neighbor models with connection details
        """
        if graph is None:
            # Build network for top models to keep it manageable
            top_models = self.get_top_models_by_field(n=1000)
            model_ids = [mid for mid, _ in top_models]
            graph = self.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
        
        model_id_str = str(model_id)
        if model_id_str not in graph:
            return []
        
        neighbors = []
        for neighbor_id in graph.neighbors(model_id_str):
            edge_data = graph.get_edge_data(model_id_str, neighbor_id, {})
            weight = edge_data.get('weight', 1.0)
            
            if weight >= min_weight:
                if neighbor_id in self.df.index:
                    row = self.df.loc[neighbor_id]
                    neighbors.append({
                        'model_id': neighbor_id,
                        'title': self._format_title(neighbor_id),
                        'author': self._extract_author(neighbor_id),
                        'weight': float(weight),
                        'library_name': str(row.get('library_name', '')) if pd.notna(row.get('library_name')) else '',
                        'pipeline_tag': str(row.get('pipeline_tag', '')) if pd.notna(row.get('pipeline_tag')) else '',
                        'downloads': int(row.get('downloads', 0)),
                        'likes': int(row.get('likes', 0))
                    })
        
        # Sort by weight (strongest connections first)
        neighbors.sort(key=lambda x: x['weight'], reverse=True)
        return neighbors[:max_neighbors]
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        graph: Optional[nx.Graph] = None,
        max_path_length: int = 5
    ) -> Optional[List[str]]:
        """
        Find shortest path between two models (graph-based search).
        Similar to graph database path queries.
        
        Args:
            source_id: Source model ID
            target_id: Target model ID
            graph: Pre-built network graph (if None, builds one)
            max_path_length: Maximum path length to search
            
        Returns:
            List of model IDs representing the path, or None if no path exists
        """
        if graph is None:
            top_models = self.get_top_models_by_field(n=1000)
            model_ids = [mid for mid, _ in top_models]
            graph = self.build_cooccurrence_network(model_ids, cooccurrence_method='combined')
        
        source_str = str(source_id)
        target_str = str(target_id)
        
        if source_str not in graph or target_str not in graph:
            return None
        
        try:
            path = nx.shortest_path(graph, source_str, target_str)
            if len(path) > max_path_length + 1:
                return None
            return path
        except nx.NetworkXNoPath:
            return None
    
    def search_by_cooccurrence(
        self,
        query_model_id: str,
        graph: Optional[nx.Graph] = None,
        max_results: int = 20,
        min_weight: float = 1.0
    ) -> List[Dict]:
        """
        Search for models that co-occur with a query model.
        Similar to graph database queries for co-assignment patterns.
        
        Args:
            query_model_id: Model to search around
            graph: Pre-built network graph (if None, builds one)
            max_results: Maximum number of results
            min_weight: Minimum co-occurrence weight
            
        Returns:
            List of co-occurring models sorted by connection strength
        """
        return self.find_neighbors(query_model_id, graph, max_results, min_weight)
    
    def search_graph_aware(
        self,
        query: str,
        graph: Optional[nx.Graph] = None,
        max_results: int = 20,
        include_neighbors: bool = True,
        neighbor_weight: float = 0.5
    ) -> List[Dict]:
        """
        Graph-aware search: finds models matching query and includes their neighbors.
        Combines text search with network relationships.
        
        Args:
            query: Text query to search for
            graph: Pre-built network graph (if None, builds one)
            max_results: Maximum number of results
            include_neighbors: Whether to include neighbors of matching models
            neighbor_weight: Weight factor for neighbor results (0-1)
            
        Returns:
            List of matching models with network context
        """
        query_lower = query.lower()
        
        # First, find direct matches
        matches = []
        for model_id, row in self.df.iterrows():
            model_id_str = str(model_id)
            model_id_lower = model_id_str.lower()
            tags = str(row.get('tags', '')).lower()
            library = str(row.get('library_name', '')).lower()
            pipeline = str(row.get('pipeline_tag', '')).lower()
            
            score = 0.0
            if query_lower in model_id_lower:
                score += 2.0  # Higher weight for model ID matches
            if query_lower in tags:
                score += 1.0
            if query_lower in library:
                score += 0.5
            if query_lower in pipeline:
                score += 0.5
            
            if score > 0:
                matches.append({
                    'model_id': model_id_str,
                    'title': self._format_title(model_id_str),
                    'author': self._extract_author(model_id_str),
                    'score': score,
                    'library_name': str(row.get('library_name', '')) if pd.notna(row.get('library_name')) else '',
                    'pipeline_tag': str(row.get('pipeline_tag', '')) if pd.notna(row.get('pipeline_tag')) else '',
                    'downloads': int(row.get('downloads', 0)),
                    'likes': int(row.get('likes', 0)),
                    'match_type': 'direct'
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Add neighbors if requested
        if include_neighbors and graph is not None:
            neighbor_results = []
            for match in matches[:10]:  # Only consider top 10 matches for neighbors
                neighbors = self.find_neighbors(match['model_id'], graph, max_neighbors=5, min_weight=1.0)
                for neighbor in neighbors:
                    # Check if already in matches
                    if not any(m['model_id'] == neighbor['model_id'] for m in matches):
                        neighbor_results.append({
                            **neighbor,
                            'score': neighbor['weight'] * neighbor_weight,
                            'match_type': 'neighbor',
                            'connected_to': match['model_id']
                        })
            
            # Combine and re-sort
            all_results = matches + neighbor_results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results[:max_results]
        
        return matches[:max_results]
    
    def find_models_by_relationship(
        self,
        model_id: str,
        relationship_type: str = 'all',
        max_results: int = 50
    ) -> List[Dict]:
        """
        Find models by specific relationship types (family, library, pipeline, tags).
        Similar to graph database relationship queries.
        
        Args:
            model_id: Model to find relationships for
            relationship_type: Type of relationship ('family', 'library', 'pipeline', 'tags', 'all')
            max_results: Maximum number of results
            
        Returns:
            List of related models
        """
        if model_id not in self.df.index:
            return []
        
        row = self.df.loc[model_id]
        related_models = []
        
        if relationship_type in ['family', 'all']:
            # Find siblings (same parent)
            parent_id = row.get('parent_model')
            if parent_id and pd.notna(parent_id):
                siblings = self.df[self.df.get('parent_model', '') == parent_id]
                for sibling_id, sibling_row in siblings.iterrows():
                    if str(sibling_id) != str(model_id):
                        related_models.append({
                            'model_id': str(sibling_id),
                            'title': self._format_title(str(sibling_id)),
                            'relationship': 'sibling',
                            'library_name': str(sibling_row.get('library_name', '')) if pd.notna(sibling_row.get('library_name')) else '',
                            'downloads': int(sibling_row.get('downloads', 0))
                        })
            
            # Find children
            children = self.df[self.df.get('parent_model', '') == model_id]
            for child_id, child_row in children.iterrows():
                related_models.append({
                    'model_id': str(child_id),
                    'title': self._format_title(str(child_id)),
                    'relationship': 'child',
                    'library_name': str(child_row.get('library_name', '')) if pd.notna(child_row.get('library_name')) else '',
                    'downloads': int(child_row.get('downloads', 0))
                })
        
        if relationship_type in ['library', 'all']:
            library = row.get('library_name')
            if library and pd.notna(library):
                same_library = self.df[
                    (self.df.get('library_name', '') == library) &
                    (self.df.index != model_id)
                ]
                for lib_model_id, lib_row in same_library.head(max_results).iterrows():
                    related_models.append({
                        'model_id': str(lib_model_id),
                        'title': self._format_title(str(lib_model_id)),
                        'relationship': 'same_library',
                        'library_name': str(library),
                        'downloads': int(lib_row.get('downloads', 0))
                    })
        
        if relationship_type in ['pipeline', 'all']:
            pipeline = row.get('pipeline_tag')
            if pipeline and pd.notna(pipeline):
                same_pipeline = self.df[
                    (self.df.get('pipeline_tag', '') == pipeline) &
                    (self.df.index != model_id)
                ]
                for pipe_model_id, pipe_row in same_pipeline.head(max_results).iterrows():
                    related_models.append({
                        'model_id': str(pipe_model_id),
                        'title': self._format_title(str(pipe_model_id)),
                        'relationship': 'same_pipeline',
                        'pipeline_tag': str(pipeline),
                        'downloads': int(pipe_row.get('downloads', 0))
                    })
        
        if relationship_type in ['tags', 'all']:
            tags = str(row.get('tags', '')).lower().split(',')
            tags = [t.strip() for t in tags if t.strip()]
            if tags:
                for other_id, other_row in self.df.iterrows():
                    if str(other_id) == str(model_id):
                        continue
                    other_tags = str(other_row.get('tags', '')).lower().split(',')
                    other_tags = [t.strip() for t in other_tags if t.strip()]
                    overlap = set(tags) & set(other_tags)
                    if len(overlap) > 0:
                        related_models.append({
                            'model_id': str(other_id),
                            'title': self._format_title(str(other_id)),
                            'relationship': 'shared_tags',
                            'shared_tags': list(overlap),
                            'downloads': int(other_row.get('downloads', 0))
                        })
        
        # Remove duplicates and sort by downloads
        seen = set()
        unique_models = []
        for model in related_models:
            if model['model_id'] not in seen:
                seen.add(model['model_id'])
                unique_models.append(model)
        
        unique_models.sort(key=lambda x: x.get('downloads', 0), reverse=True)
        return unique_models[:max_results]

