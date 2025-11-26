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
        
    def load_data(self, sample_size: Optional[int] = None, split: str = "train") -> pd.DataFrame:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            sample_size: If provided, randomly sample this many rows
            split: Dataset split to load
            
        Returns:
            DataFrame with model data
        """
        dataset = load_dataset(self.dataset_name, split=split)
        
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        self.df = dataset.to_pandas()
        
        return self.df
    
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
        
        df['combined_text'] = (
            df.get('tags', '').astype(str) + ' ' +
            df.get('pipeline_tag', '').astype(str) + ' ' +
            df.get('library_name', '').astype(str) + ' ' +
            df['modelCard'].astype(str).str[:500]
        )
        
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

