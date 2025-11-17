"""
Clustering for model grouping in latent space.
Inspired by Aella Data Explorer's K-Means clustering with silhouette optimization.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Optional
import pickle
import os


class ModelClusterer:
    """Cluster models in latent space using K-Means with automatic optimization."""
    
    def __init__(self, n_clusters_range: Tuple[int, int] = (20, 60), random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            n_clusters_range: Range of cluster counts to test (min, max)
            random_state: Random state for reproducibility
        """
        self.n_clusters_range = n_clusters_range
        self.random_state = random_state
        self.optimal_n_clusters: Optional[int] = None
        self.kmeans: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.silhouette_scores: List[float] = []
    
    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        sample_size: Optional[int] = None
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.
        Similar to Aella's approach of testing 20-60 clusters.
        
        Args:
            embeddings: 2D embeddings (n_samples, 2) from UMAP
            sample_size: If provided, sample this many points for faster computation
            
        Returns:
            Optimal number of clusters
        """
        
        # Sample for faster computation if dataset is large
        if sample_size and len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
            indices = np.arange(len(embeddings))
        
        min_clusters, max_clusters = self.n_clusters_range
        best_score = -1
        best_n = min_clusters
        
        # Test different cluster counts
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(sample_embeddings)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(sample_embeddings, labels)
                self.silhouette_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_n = n_clusters
                
        
        self.optimal_n_clusters = best_n
        return best_n
    
    def fit(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Fit K-Means clustering to embeddings.
        
        Args:
            embeddings: 2D embeddings (n_samples, 2)
            n_clusters: Number of clusters (if None, uses optimal)
            
        Returns:
            Cluster labels for each point
        """
        if n_clusters is None:
            if self.optimal_n_clusters is None:
                n_clusters = self.find_optimal_clusters(embeddings)
            else:
                n_clusters = self.optimal_n_clusters
        else:
            self.optimal_n_clusters = n_clusters
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.labels_ = self.kmeans.fit_predict(embeddings)
        
        return self.labels_
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if self.kmeans is None:
            raise ValueError("Must fit clusterer before predicting")
        return self.kmeans.predict(embeddings)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster center coordinates."""
        if self.kmeans is None:
            raise ValueError("Must fit clusterer before getting centers")
        return self.kmeans.cluster_centers_
    
    def save(self, filepath: str):
        """Save clusterer to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'labels_': self.labels_,
                'optimal_n_clusters': self.optimal_n_clusters,
                'silhouette_scores': self.silhouette_scores
            }, f)
    
    def load(self, filepath: str):
        """Load clusterer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.labels_ = data['labels_']
            self.optimal_n_clusters = data['optimal_n_clusters']
            self.silhouette_scores = data.get('silhouette_scores', [])


def generate_cluster_labels_tfidf(
    model_ids: List[str],
    labels: np.ndarray,
    n_words: int = 3
) -> dict:
    """
    Generate cluster labels using TF-IDF analysis of model IDs.
    Similar to Aella's TF-IDF approach for initial cluster labels.
    
    Args:
        model_ids: List of model IDs
        labels: Cluster labels
        n_words: Number of top words to use in label
        
    Returns:
        Dictionary mapping cluster_id -> label string
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    unique_labels = np.unique(labels)
    cluster_labels = {}
    
    for cluster_id in unique_labels:
        # Get model IDs in this cluster
        cluster_models = [model_ids[i] for i in range(len(model_ids)) if labels[i] == cluster_id]
        
        if len(cluster_models) == 0:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Extract words from model IDs (split by common separators)
        words = []
        for model_id in cluster_models:
            # Split by common separators
            parts = model_id.replace('-', ' ').replace('_', ' ').replace('/', ' ').split()
            words.extend(parts)
        
        if len(words) == 0:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Use TF-IDF to find most distinctive words
        try:
            vectorizer = TfidfVectorizer(max_features=n_words, stop_words='english')
            # Create documents: one per model
            docs = [' '.join(model_id.replace('-', ' ').replace('_', ' ').split()) for model_id in cluster_models]
            if len(docs) > 0:
                tfidf_matrix = vectorizer.fit_transform(docs)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top words by average TF-IDF score
                scores = tfidf_matrix.mean(axis=0).A1
                top_indices = scores.argsort()[-n_words:][::-1]
                top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                if top_words:
                    label = ' '.join(top_words[:n_words]).title()
                    cluster_labels[cluster_id] = label
                else:
                    cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
        except Exception as e:
            # Error generating TF-IDF label for cluster
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    
    return cluster_labels

