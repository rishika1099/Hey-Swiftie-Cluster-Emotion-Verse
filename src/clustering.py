"""
CLUSTERING MODULE
=================
This module groups Taylor Swift songs into clusters based on emotional similarity.

KEY CONCEPTS:
- Unsupervised Learning: Finding patterns without labeled data
- Clustering: Grouping similar items together
- Dimensionality Reduction: Visualizing high-dimensional data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

class SongClusterer:
    """
    Clusters songs based on emotional and musical features.
    
    CONCEPT: What is Clustering?
    - Imagine sorting songs into piles based on similarity
    - Songs in the same pile should feel similar emotionally
    - We don't tell the algorithm what the categories are - it finds them!
    - This is "unsupervised learning" - no labels needed
    """
    
    def __init__(self):
        """
        Initialize the clusterer.
        """
        self.scaler = None       # Will normalize features
        self.model = None        # Will be our clustering algorithm
        self.feature_names = []  # Track which features we're using
        self.cluster_labels = {} # Human-readable names for clusters
        
    def prepare_features(self, df, feature_columns=None):
        """
        Prepare features for clustering.
        
        Args:
            df (pd.DataFrame): DataFrame with emotion features
            feature_columns (list): Which columns to use for clustering
            
        Returns:
            np.array: Normalized feature matrix
            
        CONCEPT: Feature Selection
        - Not all data is useful for clustering
        - We pick features that capture emotional/thematic differences
        - More features isn't always better (curse of dimensionality!)
        """
        
        if feature_columns is None:
            # Default: Use emotion scores + compound features
            feature_columns = [
                # Basic emotions
                'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
                # Compound features
                'valence', 'arousal', 'dominance', 'emotional_complexity',
                # Lyrical features
                'word_count'
            ]
        
        self.feature_names = feature_columns
        print(f"\n=== Preparing Features for Clustering ===")
        print(f"Using {len(feature_columns)} features:")
        for feat in feature_columns:
            print(f"  - {feat}")
        
        # Extract feature matrix
        # CONCEPT: Feature Matrix
        # - Each row = one song
        # - Each column = one feature (emotion score, valence, etc.)
        # - This is the "X" in machine learning
        X = df[feature_columns].values
        
        # Handle any missing values
        # CONCEPT: Missing data can break algorithms
        # We replace NaN with column mean (imputation)
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # Standardize features
        # CONCEPT: Feature Scaling/Normalization
        # Problem: 'word_count' might range 50-500, while 'joy' ranges 0-1
        # Solution: Scale all features to have mean=0, std=1
        # Why? Clustering algorithms care about distances - large-scale features dominate!
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nFeature matrix shape: {X_scaled.shape}")
        print(f"(That's {X_scaled.shape[0]} songs × {X_scaled.shape[1]} features)")
        
        return X_scaled
    
    def find_optimal_clusters(self, X, min_k=3, max_k=15):
        """
        Find the optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X (np.array): Scaled feature matrix
            min_k (int): Minimum number of clusters to try
            max_k (int): Maximum number of clusters to try
            
        Returns:
            int: Suggested optimal number of clusters
            
        CONCEPT: How Many Clusters?
        - Too few clusters: Songs are too different within clusters
        - Too many clusters: Clusters are too similar to each other
        - We use metrics to find the "sweet spot"
        """
        print(f"\n=== Finding Optimal Number of Clusters ===")
        print(f"Testing k from {min_k} to {max_k}...")
        
        inertias = []        # Within-cluster sum of squares
        silhouettes = []     # How well-separated are clusters?
        db_scores = []       # Davies-Bouldin score (lower is better)
        
        for k in range(min_k, max_k + 1):
            # CONCEPT: K-Means Algorithm
            # 1. Randomly place k "centroids" (cluster centers)
            # 2. Assign each song to nearest centroid
            # 3. Move centroids to average of their assigned songs
            # 4. Repeat steps 2-3 until centroids stop moving
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,      # For reproducibility
                n_init=10,            # Try 10 different initializations
                max_iter=300          # Maximum iterations
            )
            labels = kmeans.fit_predict(X)
            
            # Metric 1: Inertia (lower is better)
            # CONCEPT: How compact are clusters?
            # Sum of squared distances from songs to their cluster center
            inertias.append(kmeans.inertia_)
            
            # Metric 2: Silhouette Score (higher is better, range: -1 to 1)
            # CONCEPT: How well-separated are clusters?
            # Close to 1: Songs are far from other clusters
            # Close to 0: Songs are on cluster boundaries
            # Negative: Songs might be in wrong cluster
            sil_score = silhouette_score(X, labels)
            silhouettes.append(sil_score)
            
            # Metric 3: Davies-Bouldin Score (lower is better)
            # CONCEPT: Ratio of within-cluster to between-cluster distances
            db_score = davies_bouldin_score(X, labels)
            db_scores.append(db_score)
            
            print(f"k={k}: Silhouette={sil_score:.3f}, DB Score={db_score:.3f}")
        
        # Plot metrics
        self._plot_clustering_metrics(
            range(min_k, max_k + 1),
            inertias,
            silhouettes,
            db_scores
        )
        
        # Find optimal k using silhouette score
        # CONCEPT: Automated decision
        # We pick k with highest silhouette score
        optimal_k = min_k + np.argmax(silhouettes)
        print(f"\n✓ Suggested optimal k: {optimal_k}")
        print(f"  (Based on highest silhouette score: {max(silhouettes):.3f})")
        
        return optimal_k
    
    def _plot_clustering_metrics(self, k_range, inertias, silhouettes, db_scores):
        """
        Visualize clustering metrics to help choose optimal k.
        
        CONCEPT: Data Visualization
        - Numbers alone can be hard to interpret
        - Plots help us see trends and make decisions
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Elbow curve (inertia)
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        # Plot 2: Silhouette score
        axes[1].plot(k_range, silhouettes, 'go-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score (higher is better)')
        axes[1].grid(True)
        
        # Plot 3: Davies-Bouldin score
        axes[2].plot(k_range, db_scores, 'ro-')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Score (lower is better)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('data/processed/clustering_metrics.png', dpi=150)
        print("\n✓ Saved clustering metrics plot to: data/processed/clustering_metrics.png")
        plt.close()
    
    def fit_kmeans(self, X, n_clusters):
        """
        Fit K-Means clustering model.
        
        Args:
            X (np.array): Scaled feature matrix
            n_clusters (int): Number of clusters
            
        Returns:
            np.array: Cluster labels for each song
        """
        print(f"\n=== Fitting K-Means with {n_clusters} clusters ===")
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,        # More initializations for better results
            max_iter=500      # More iterations to ensure convergence
        )
        
        # Fit the model and get cluster assignments
        # CONCEPT: fit_predict does two things:
        # 1. Learn the cluster centers (fit)
        # 2. Assign each song to a cluster (predict)
        labels = self.model.fit_predict(X)
        
        print(f"✓ Clustering complete!")
        print(f"Cluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} songs ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def visualize_clusters(self, X, labels, df, save_path='data/processed/cluster_visualization.png'):
        """
        Visualize clusters in 2D using PCA.
        
        Args:
            X (np.array): Scaled feature matrix
            labels (np.array): Cluster assignments
            df (pd.DataFrame): Original DataFrame (for metadata)
            save_path (str): Where to save the plot
            
        CONCEPT: Dimensionality Reduction
        - We have 10+ dimensions (features), but can only visualize 2D/3D
        - PCA (Principal Component Analysis) finds the "best" 2D projection
        - "Best" = preserves as much variance as possible
        - Think of it like finding the best angle to photograph a 3D object
        """
        print("\n=== Visualizing Clusters ===")
        
        # Apply PCA to reduce to 2 dimensions
        # CONCEPT: PCA finds new axes (principal components)
        # PC1 = direction with most variance in data
        # PC2 = direction with second-most variance (perpendicular to PC1)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # How much variance is explained by these 2 dimensions?
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"2D projection explains {variance_explained*100:.1f}% of variance")
        print("(Higher is better - means we're not losing too much information)")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster with different color
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            # Get songs in this cluster
            mask = labels == cluster_id
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[i]],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=50
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title('Taylor Swift Songs - Cluster Visualization (2D PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
        plt.close()
    
    def analyze_clusters(self, df, labels):
        """
        Analyze what makes each cluster unique.
        
        Args:
            df (pd.DataFrame): DataFrame with features and metadata
            labels (np.array): Cluster assignments
            
        CONCEPT: Cluster Interpretation
        - Clustering is unsupervised - it finds groups but doesn't name them
        - We need to analyze what each cluster represents
        - Look at: average emotions, representative songs, eras
        """
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60)
        
        # Add cluster labels to DataFrame
        df['cluster'] = labels
        
        # Analyze each cluster
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            
            print(f"\n{'='*60}")
            print(f"CLUSTER {cluster_id} ({len(cluster_df)} songs)")
            print(f"{'='*60}")
            
            # Average emotion scores
            print("\n--- Average Emotions ---")
            emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
            avg_emotions = cluster_df[emotion_cols].mean().sort_values(ascending=False)
            for emotion, score in avg_emotions.items():
                print(f"  {emotion.capitalize()}: {score:.3f}")
            
            # Dominant emotion distribution
            print("\n--- Dominant Emotions ---")
            dom_emotions = cluster_df['dominant_emotion'].value_counts()
            for emotion, count in dom_emotions.items():
                pct = count / len(cluster_df) * 100
                print(f"  {emotion}: {count} songs ({pct:.1f}%)")
            
            # Compound features
            print("\n--- Average Characteristics ---")
            print(f"  Valence (positivity): {cluster_df['valence'].mean():.3f}")
            print(f"  Arousal (intensity): {cluster_df['arousal'].mean():.3f}")
            print(f"  Dominance (empowerment): {cluster_df['dominance'].mean():.3f}")
            print(f"  Emotional complexity: {cluster_df['emotional_complexity'].mean():.3f}")
            
            # Era distribution
            print("\n--- Era Distribution ---")
            era_counts = cluster_df['era'].value_counts().head(5)
            for era, count in era_counts.items():
                pct = count / len(cluster_df) * 100
                print(f"  {era}: {count} songs ({pct:.1f}%)")
            
            # Sample songs
            print("\n--- Representative Songs ---")
            # Get songs closest to cluster center
            if hasattr(self.model, 'cluster_centers_'):
                # Calculate distance to cluster center
                center = self.model.cluster_centers_[cluster_id]
                X_cluster = self.scaler.transform(cluster_df[self.feature_names].values)
                distances = np.linalg.norm(X_cluster - center, axis=1)
                closest_indices = np.argsort(distances)[:5]
                
                for idx in closest_indices:
                    song = cluster_df.iloc[idx]
                    print(f"  • {song['track_title']} ({song['album_name']})")
        
        return df
    
    def assign_cluster_labels(self, df):
        """
        Manually assign meaningful labels to clusters based on analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with cluster assignments
            
        Returns:
            pd.DataFrame: DataFrame with 'cluster_label' column
            
        CONCEPT: Human-in-the-Loop
        - While algorithms find patterns, humans interpret meaning
        - After analyzing clusters, we give them descriptive names
        - This makes the system more interpretable for users
        """
        print("\n=== Assigning Cluster Labels ===")
        print("Based on the cluster analysis above, manually assign labels.")
        print("Example labels: 'Heartbreak Ballads', 'Empowerment Anthems', etc.")
        
        # This is a placeholder - you'll customize based on your results
        # Create specific labels based on the actual cluster analysis
        # These are derived from analyzing the emotional patterns in each cluster
        self.cluster_labels = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            
            # Get cluster characteristics
            avg_valence = cluster_df['valence'].mean()
            avg_arousal = cluster_df['arousal'].mean()
            avg_dominance = cluster_df['dominance'].mean()
            top_emotion = cluster_df['dominant_emotion'].mode()[0]
            
            # Get emotion distribution
            emotion_counts = cluster_df['dominant_emotion'].value_counts()
            primary_emotion_pct = emotion_counts.iloc[0] / len(cluster_df) if len(emotion_counts) > 0 else 0
            
            # Assign specific labels based on characteristics
            # Pattern: dominant emotion + intensity/mood descriptor
            
            if primary_emotion_pct > 0.9:
                # Very pure emotion clusters (>90% one emotion)
                if top_emotion == 'sadness':
                    label = "Heartbreak Ballads"
                elif top_emotion == 'joy':
                    label = "Happy & Celebratory"
                elif top_emotion == 'fear' and avg_arousal > 0.6:
                    label = "Intense & Vulnerable"
                elif top_emotion == 'anger':
                    label = "Fierce & Empowered"
                elif top_emotion == 'surprise':
                    label = "Uplifting & Surprising"
                else:
                    label = f"Pure {top_emotion.title()}"
            
            elif top_emotion == 'neutral' and avg_arousal < 0.4:
                # Calm, reflective songs
                label = "Introspective & Calm"
            
            elif avg_valence < 0.3 and avg_arousal < 0.4:
                # Low energy, sad
                label = "Melancholic & Slow"
            
            elif avg_valence < 0.4 and avg_arousal > 0.5:
                # High energy, negative
                if avg_dominance < 0.3:
                    label = "Intense & Dark"
                else:
                    label = "Edgy & Powerful"
            
            elif avg_valence > 0.7:
                # Positive vibes
                if avg_arousal > 0.6:
                    label = "Energetic & Bright"
                else:
                    label = "Peaceful & Content"
            
            else:
                # Mixed/complex emotions
                if 'disgust' in emotion_counts.index[:3] or 'anger' in emotion_counts.index[:3]:
                    label = "Dark & Complex"
                else:
                    label = "Bittersweet & Reflective"
            
            self.cluster_labels[cluster_id] = label
            print(f"  Cluster {cluster_id}: '{label}'")
        
        # Add labels to DataFrame
        df['cluster_label'] = df['cluster'].map(self.cluster_labels)
        
        return df
    
    def save_model(self, output_dir='models'):
        """
        Save the clustering model and scaler.
        
        CONCEPT: Model Persistence
        - Training takes time, so we save the model
        - Later, we can load it without retraining
        - Essential for deployment!
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'clustering_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = output_dir / 'feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names and cluster labels
        metadata = {
            'feature_names': self.feature_names,
            'cluster_labels': self.cluster_labels
        }
        metadata_path = output_dir / 'clustering_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✓ Saved model to: {output_dir}")
        print(f"  - {model_path.name}")
        print(f"  - {scaler_path.name}")
        print(f"  - {metadata_path.name}")


# Example usage
if __name__ == "__main__":
    """
    Run the complete clustering pipeline.
    """
    
    # Load data with emotions
    print("Loading data with emotion features...")
    df = pd.read_csv('data/processed/taylor_swift_with_emotions.csv')
    
    # Initialize clusterer
    clusterer = SongClusterer()
    
    # Prepare features
    X = clusterer.prepare_features(df)
    
    # Find optimal number of clusters
    optimal_k = clusterer.find_optimal_clusters(X, min_k=4, max_k=12)
    
    # Or manually choose based on the analysis
    # optimal_k = 7  # Uncomment to override
    
    # Fit clustering model
    labels = clusterer.fit_kmeans(X, n_clusters=optimal_k)
    
    # Visualize
    clusterer.visualize_clusters(X, labels, df)
    
    # Analyze clusters
    df = clusterer.analyze_clusters(df, labels)
    
    # Assign meaningful labels
    df = clusterer.assign_cluster_labels(df)
    
    # Save everything
    clusterer.save_model()
    df.to_csv('data/processed/taylor_swift_clustered.csv', index=False)
    
    print("\n" + "="*60)
    print("✓ CLUSTERING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the cluster analysis above")
    print("2. Manually refine cluster labels in assign_cluster_labels()")
    print("3. Use these clusters for diary entry matching!")
