"""
DIARY CLASSIFIER MODULE
=======================
This module matches diary entries to Taylor Swift song clusters.

KEY CONCEPTS:
- Transfer Learning: Using models trained on songs for diary entries
- Semantic Similarity: Finding which cluster "feels" most like the diary entry
- Classification: Assigning a label (cluster) to new text
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class DiaryClassifier:
    """
    Classifies diary entries to song clusters.
    
    CONCEPT: How This Works
    1. User writes a diary entry
    2. We extract emotions from it (same way we did for songs)
    3. We find which cluster has similar emotional profile
    4. We return that cluster's songs and theme
    
    This is "nearest neighbor" classification in emotion space!
    """
    
    def __init__(self, clustered_data_path, model_dir='models'):
        """
        Initialize the classifier with pre-trained models.
        
        Args:
            clustered_data_path (str): Path to clustered songs CSV
            model_dir (str): Directory with saved models
            
        CONCEPT: Loading Pre-trained Models
        - We don't retrain - we load models we already trained
        - This is much faster and allows deployment
        """
        print("=== Loading Diary Classifier ===")
        
        model_dir = Path(model_dir)
        
        # Load clustered songs data
        # CONCEPT: Reference Database
        # We need the original songs to recommend from each cluster
        self.songs_df = pd.read_csv(clustered_data_path)
        print(f"✓ Loaded {len(self.songs_df)} songs with clusters")
        
        # Load clustering model and scaler
        # CONCEPT: These were saved after training
        # Model knows the cluster centers
        # Scaler normalizes features the same way
        self.model = joblib.load(model_dir / 'clustering_model.pkl')
        self.scaler = joblib.load(model_dir / 'feature_scaler.pkl')
        
        # Load metadata (feature names, cluster labels)
        metadata = joblib.load(model_dir / 'clustering_metadata.pkl')
        self.feature_names = metadata['feature_names']
        self.cluster_labels = metadata['cluster_labels']
        
        print(f"✓ Loaded clustering model with {len(self.cluster_labels)} clusters")
        print("\nCluster labels:")
        for cluster_id, label in self.cluster_labels.items():
            print(f"  {cluster_id}: {label}")
    
    def extract_diary_features(self, diary_text, emotion_analyzer):
        """
        Extract emotional features from diary entry.
        
        Args:
            diary_text (str): User's diary entry
            emotion_analyzer: SentimentAnalyzer instance
            
        Returns:
            np.array: Feature vector (1 x num_features)
            
        CONCEPT: Feature Extraction
        - We analyze the diary the same way we analyzed songs
        - This ensures features are comparable
        - Same features → same emotional "space"
        """
        # Get emotion scores
        emotions = emotion_analyzer.analyze_emotion(diary_text)
        
        # Create feature dictionary
        # CONCEPT: We need to match the exact features used in training
        features = {
            'anger': emotions['anger'],
            'disgust': emotions['disgust'],
            'fear': emotions['fear'],
            'joy': emotions['joy'],
            'sadness': emotions['sadness'],
            'surprise': emotions['surprise']
        }
        
        # Calculate compound features (same formulas as in training)
        # CONCEPT: Feature Engineering Must Be Consistent
        # If we trained with valence, arousal, etc., we need them here too
        features['valence'] = (
            emotions['joy'] * 1.0 +
            emotions['surprise'] * 0.5 +
            emotions['neutral'] * 0.0 +
            emotions['sadness'] * -1.0 +
            emotions['anger'] * -1.0 +
            emotions['fear'] * -0.8 +
            emotions['disgust'] * -0.8
        )
        
        features['arousal'] = (
            emotions['anger'] * 1.0 +
            emotions['joy'] * 0.8 +
            emotions['surprise'] * 0.9 +
            emotions['fear'] * 0.7 +
            emotions['sadness'] * 0.3 +
            emotions['neutral'] * 0.0
        )
        
        features['dominance'] = (
            emotions['anger'] * 0.8 +
            emotions['joy'] * 0.6 +
            emotions['sadness'] * -0.7 +
            emotions['fear'] * -0.9 +
            emotions['disgust'] * 0.2
        )
        
        # Emotional complexity
        emotion_scores = [emotions[e] for e in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']]
        features['emotional_complexity'] = np.std(emotion_scores)
        
        # Word count (approximate)
        features['word_count'] = len(diary_text.split())
        
        # Convert to array in correct order
        # CONCEPT: Order Matters!
        # Features must be in the exact same order as training
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # Reshape to 2D array (required by sklearn)
        # CONCEPT: Shape = (1, num_features) means "1 sample with multiple features"
        feature_vector = feature_vector.reshape(1, -1)
        
        return feature_vector, emotions
    
    def classify_diary(self, diary_text, emotion_analyzer):
        """
        Classify diary entry to a cluster.
        
        Args:
            diary_text (str): User's diary entry
            emotion_analyzer: SentimentAnalyzer instance
            
        Returns:
            dict: Classification results
            
        CONCEPT: The Classification Pipeline
        1. Extract features from diary
        2. Normalize features (using saved scaler)
        3. Find nearest cluster center
        4. Return cluster info + recommended songs
        """
        print("\n=== Classifying Diary Entry ===")
        
        # Extract and normalize features
        features, raw_emotions = self.extract_diary_features(diary_text, emotion_analyzer)
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        # CONCEPT: predict() finds the nearest cluster center
        # This is like asking "which song group feels most like this diary?"
        cluster_id = self.model.predict(features_scaled)[0]
        cluster_label = self.cluster_labels[cluster_id]
        
        print(f"✓ Matched to: Cluster {cluster_id} - '{cluster_label}'")
        
        # Get emotional profile
        print("\n--- Detected Emotions ---")
        sorted_emotions = sorted(raw_emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions[:3]:
            if emotion != 'neutral':
                print(f"  {emotion.capitalize()}: {score:.3f}")
        
        # Get songs from this cluster
        cluster_songs = self.songs_df[self.songs_df['cluster'] == cluster_id]
        
        # Calculate similarity scores
        # CONCEPT: Distance in Feature Space
        # Songs closer to the diary entry are more similar
        # We use Euclidean distance (like measuring physical distance)
        cluster_center = self.model.cluster_centers_[cluster_id]
        song_features = self.scaler.transform(
            cluster_songs[self.feature_names].values
        )
        
        # Calculate distance from diary to each song
        distances = np.linalg.norm(song_features - features_scaled, axis=1)
        
        # Convert distance to similarity (higher = more similar)
        # CONCEPT: Inverse distance → songs with smaller distance = higher similarity
        max_distance = distances.max()
        similarities = 1 - (distances / max_distance)
        
        # Add similarity scores to cluster songs
        cluster_songs = cluster_songs.copy()
        cluster_songs['similarity'] = similarities
        
        # Get top 10 most similar songs
        top_songs = cluster_songs.nlargest(10, 'similarity')
        
        # Prepare result
        result = {
            'cluster_id': int(cluster_id),
            'cluster_label': cluster_label,
            'emotions': raw_emotions,
            'top_emotion': max(raw_emotions.items(), key=lambda x: x[1])[0],
            'cluster_size': len(cluster_songs),
            'recommended_songs': top_songs[[
                'track_title', 'album_name', 'similarity', 
                'valence', 'arousal', 'dominant_emotion'
            ]].to_dict('records')
        }
        
        return result
    
    def get_cluster_stats(self, cluster_id):
        """
        Get detailed statistics for a cluster.
        
        Args:
            cluster_id (int): Cluster ID
            
        Returns:
            dict: Cluster statistics
            
        CONCEPT: Understanding Cluster Characteristics
        - Each cluster has a unique emotional "personality"
        - We provide this info to help generate appropriate letters
        """
        cluster_songs = self.songs_df[self.songs_df['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': int(cluster_id),
            'cluster_label': self.cluster_labels[cluster_id],
            'num_songs': len(cluster_songs),
            'avg_valence': float(cluster_songs['valence'].mean()),
            'avg_arousal': float(cluster_songs['arousal'].mean()),
            'avg_dominance': float(cluster_songs['dominance'].mean()),
            'dominant_emotions': cluster_songs['dominant_emotion'].value_counts().head(3).to_dict(),
            'top_albums': cluster_songs['album_name'].value_counts().head(5).to_dict(),
            'representative_songs': cluster_songs.nsmallest(5, 'emotional_complexity')[[
                'track_title', 'album_name'
            ]].to_dict('records')
        }
        
        return stats
    
    def get_all_clusters_summary(self):
        """
        Get summary of all clusters for display.
        
        Returns:
            list: List of cluster summaries
            
        CONCEPT: Cluster Overview
        - Helps users understand the emotional landscape
        - Useful for debugging and explanation
        """
        summaries = []
        
        for cluster_id in sorted(self.cluster_labels.keys()):
            stats = self.get_cluster_stats(cluster_id)
            
            summary = {
                'id': cluster_id,
                'label': stats['cluster_label'],
                'size': stats['num_songs'],
                'characteristics': {
                    'valence': 'positive' if stats['avg_valence'] > 0.5 else 'negative',
                    'arousal': 'high energy' if stats['avg_arousal'] > 0.5 else 'calm',
                    'dominance': 'empowered' if stats['avg_dominance'] > 0.5 else 'vulnerable'
                },
                'top_emotion': list(stats['dominant_emotions'].keys())[0],
                'sample_songs': [s['track_title'] for s in stats['representative_songs'][:3]]
            }
            
            summaries.append(summary)
        
        return summaries


# Example usage
if __name__ == "__main__":
    """
    Test the diary classifier with sample entries.
    """
    from sentiment_analyzer import SentimentAnalyzer
    
    # Initialize
    print("Initializing classifier...")
    classifier = DiaryClassifier('data/processed/taylor_swift_clustered.csv')
    emotion_analyzer = SentimentAnalyzer()
    
    # Show cluster summary
    print("\n" + "="*60)
    print("AVAILABLE CLUSTERS")
    print("="*60)
    summaries = classifier.get_all_clusters_summary()
    for summary in summaries:
        print(f"\n{summary['id']}: {summary['label']}")
        print(f"  Size: {summary['size']} songs")
        print(f"  Vibe: {summary['characteristics']['valence']}, "
              f"{summary['characteristics']['arousal']}, "
              f"{summary['characteristics']['dominance']}")
        print(f"  Top emotion: {summary['top_emotion']}")
        print(f"  Examples: {', '.join(summary['sample_songs'])}")
    
    # Test with sample diary entries
    test_entries = [
        "Today was amazing! I felt so confident and everything went perfectly.",
        "I'm so heartbroken. I thought we had something special but it's over.",
        "Feeling anxious about tomorrow. So many unknowns and I feel unprepared.",
        "Just reflecting on old memories. There's beauty in the sadness of what used to be."
    ]
    
    print("\n" + "="*60)
    print("TESTING DIARY CLASSIFICATION")
    print("="*60)
    
    for i, entry in enumerate(test_entries, 1):
        print(f"\n--- Test Entry {i} ---")
        print(f"Entry: \"{entry}\"")
        
        result = classifier.classify_diary(entry, emotion_analyzer)
        
        print(f"\nTop 3 recommended songs:")
        for j, song in enumerate(result['recommended_songs'][:3], 1):
            print(f"  {j}. {song['track_title']} - {song['album_name']}")
            print(f"     Similarity: {song['similarity']:.3f}")
        
        print("\n" + "-"*60)
    
    print("\n✓ Diary classifier testing complete!")
