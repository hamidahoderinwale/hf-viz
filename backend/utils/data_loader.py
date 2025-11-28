"""
Data loading and preprocessing for the Hugging Face model ecosystem dataset.
"""
import pandas as pd
from datasets import load_dataset
from typing import Optional, Dict, List
import numpy as np


class ModelDataLoader:
    """Load and preprocess model data from Hugging Face dataset."""
    
    def __init__(self, dataset_name: str = "modelbiome/ai_ecosystem"):
        self.dataset_name = dataset_name
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self, sample_size: Optional[int] = None, split: str = "train", 
                  prioritize_base_models: bool = True) -> pd.DataFrame:
        """
        Load dataset from Hugging Face Hub with methodological sampling.
        
        Args:
            sample_size: If provided, sample this many rows using stratified approach
            split: Dataset split to load
            prioritize_base_models: If True, prioritize base models (no parent) in sampling
            
        Returns:
            DataFrame with model data
        """
        dataset = load_dataset(self.dataset_name, split=split)
        df_full = dataset.to_pandas()
        
        if sample_size and len(df_full) > sample_size:
            if prioritize_base_models:
                # Methodological sampling: prioritize base models
                df_full = self._stratified_sample(df_full, sample_size)
            else:
                # Random sampling (old approach)
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
                df_full = dataset.to_pandas()
        
        self.df = df_full
        return self.df
    
    def _stratified_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        Stratified sampling prioritizing base models and popular models.
        
        Strategy:
        1. Include ALL base models (no parent) if they fit in sample_size
        2. Add popular models (high downloads/likes)
        3. Fill remaining with diverse models across libraries/tasks
        
        Args:
            df: Full DataFrame
            sample_size: Target sample size
            
        Returns:
            Sampled DataFrame
        """
        # Identify base models (no parent)
        # parent_model is stored as string representation of list: '[]' for base models
        base_models = df[
            df['parent_model'].isna() | 
            (df['parent_model'] == '') | 
            (df['parent_model'] == '[]') |
            (df['parent_model'] == 'null')
        ]
        
        # Start with base models
        if len(base_models) <= sample_size:
            # All base models fit - include them all
            sampled = base_models.copy()
            remaining_size = sample_size - len(sampled)
            
            # Get non-base models
            non_base = df[~df.index.isin(sampled.index)]
            
            if remaining_size > 0 and len(non_base) > 0:
                # Add popular derived models and diverse samples
                # Sort by downloads + likes for popularity
                non_base['popularity_score'] = (
                    non_base.get('downloads', 0).fillna(0) + 
                    non_base.get('likes', 0).fillna(0) * 100  # Weight likes more
                )
                
                # Take top 50% by popularity, 50% stratified by library
                popular_size = min(remaining_size // 2, len(non_base))
                diverse_size = remaining_size - popular_size
                
                # Popular models
                popular_models = non_base.nlargest(popular_size, 'popularity_score')
                sampled = pd.concat([sampled, popular_models])
                
                # Diverse sampling across libraries
                if diverse_size > 0:
                    remaining = non_base[~non_base.index.isin(popular_models.index)]
                    if len(remaining) > 0:
                        # Stratify by library if possible
                        if 'library_name' in remaining.columns:
                            libraries = remaining['library_name'].value_counts()
                            diverse_samples = []
                            per_library = max(1, diverse_size // len(libraries))
                            
                            for library in libraries.index:
                                lib_models = remaining[remaining['library_name'] == library]
                                n_sample = min(per_library, len(lib_models))
                                diverse_samples.append(lib_models.sample(n=n_sample, random_state=42))
                            
                            diverse_df = pd.concat(diverse_samples).head(diverse_size)
                        else:
                            diverse_df = remaining.sample(n=min(diverse_size, len(remaining)), random_state=42)
                        
                        sampled = pd.concat([sampled, diverse_df])
                
                sampled = sampled.drop(columns=['popularity_score'], errors='ignore')
        else:
            # Too many base models - sample from them strategically
            # Prioritize popular base models
            base_models = base_models.copy()  # Avoid SettingWithCopyWarning
            base_models['popularity_score'] = (
                base_models.get('downloads', 0).fillna(0) + 
                base_models.get('likes', 0).fillna(0) * 100
            )
            sampled = base_models.nlargest(sample_size, 'popularity_score')
            sampled = sampled.drop(columns=['popularity_score'], errors='ignore')
        
        return sampled.reset_index(drop=True)
    
    def preprocess_for_embedding(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess data for embedding generation.
        Combines text fields into a single representation.
        
        Args:
            df: DataFrame to process (uses self.df if None)
            
        Returns:
            DataFrame with combined text field
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        text_fields = ['tags', 'pipeline_tag', 'library_name', 'modelCard']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('')
        
        # Build combined text from available fields
        df['combined_text'] = (
            df.get('tags', '').astype(str) + ' ' +
            df.get('pipeline_tag', '').astype(str) + ' ' +
            df.get('library_name', '').astype(str)
        )
        
        # Add modelCard if available (only in withmodelcards dataset)
        if 'modelCard' in df.columns:
            df['combined_text'] = df['combined_text'] + ' ' + df['modelCard'].astype(str).str[:500]
        
        return df
    
    def filter_data(
        self,
        df: Optional[pd.DataFrame] = None,
        min_downloads: Optional[int] = None,
        min_likes: Optional[int] = None,
        libraries: Optional[List[str]] = None,
        pipeline_tags: Optional[List[str]] = None,
        search_query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter dataset based on criteria.
        
        Args:
            df: DataFrame to filter (uses self.df if None)
            min_downloads: Minimum download count
            min_likes: Minimum like count
            libraries: List of library names to include
            pipeline_tags: List of pipeline tags to include
            search_query: Text search in model_id or tags
            
        Returns:
            Filtered DataFrame
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        if min_downloads is not None:
            downloads_col = df.get('downloads', pd.Series([0] * len(df), index=df.index))
            df = df[downloads_col >= min_downloads]
        
        if min_likes is not None:
            likes_col = df.get('likes', pd.Series([0] * len(df), index=df.index))
            df = df[likes_col >= min_likes]
        
        if libraries:
            library_col = df.get('library_name', pd.Series([''] * len(df), index=df.index))
            df = df[library_col.isin(libraries)]
        
        if pipeline_tags:
            pipeline_col = df.get('pipeline_tag', pd.Series([''] * len(df), index=df.index))
            df = df[pipeline_col.isin(pipeline_tags)]
        
        if search_query:
            query_lower = search_query.lower()
            model_id_col = df.get('model_id', '').astype(str).str.lower()
            tags_col = df.get('tags', '').astype(str).str.lower()
            mask = model_id_col.str.contains(query_lower, na=False) | tags_col.str.contains(query_lower, na=False)
            df = df[mask]
        
        return df
    
    def get_unique_values(self, column: str) -> List[str]:
        """Get unique non-null values from a column."""
        if self.df is None:
            return []
        values = self.df[column].dropna().unique().tolist()
        return sorted([str(v) for v in values if v and str(v) != 'nan'])

