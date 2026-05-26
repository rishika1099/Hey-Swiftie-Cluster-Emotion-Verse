"""
SENTIMENT ANALYZER MODULE
=========================
This module extracts emotions and themes from Taylor Swift lyrics.

KEY CONCEPTS:
- Sentiment Analysis: Using AI to understand emotional content in text
- Pre-trained Models: Using models already trained by others
- Feature Extraction: Converting text into numerical values we can analyze
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """
    Analyzes emotional content in song lyrics.
    
    CONCEPT: Why Sentiment Analysis?
    - Computers don't "understand" emotions naturally
    - We use AI models trained on millions of texts to detect emotions
    - These models learned patterns like: "heartbroken" → sadness, "dancing" → joy
    """
    
    def __init__(self, model_name='j-hartmann/emotion-english-distilroberta-base'):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): HuggingFace model to use for emotion detection
            
        CONCEPT: Pre-trained Models
        - Training emotion models from scratch takes massive data and compute
        - We use models already trained by researchers
        - This specific model detects: anger, disgust, fear, joy, neutral, sadness, surprise
        """
        print("Loading emotion detection model...")
        print(f"Model: {model_name}")
        
        # Check if GPU is available (makes processing MUCH faster)
        # CONCEPT: GPU vs CPU
        # - GPU (Graphics Processing Unit): Designed for parallel processing, great for AI
        # - CPU: General purpose, slower for AI tasks
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"Using device: {device_name}")
        
        # Load the emotion classification pipeline
        # CONCEPT: Pipeline
        # - HuggingFace pipelines handle all the complex preprocessing/postprocessing
        # - We just pass in text and get emotions back!
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Return scores for ALL emotions, not just the top one
            device=self.device
        )
        
        print("✓ Model loaded successfully!\n")
        
    def analyze_emotion(self, text, max_length=512):
        """
        Analyze emotions in a single text.
        
        Args:
            text (str): Lyrics or diary entry to analyze
            max_length (int): Maximum text length (longer texts are truncated)
            
        Returns:
            dict: Emotion scores (e.g., {'joy': 0.8, 'sadness': 0.1, ...})
            
        CONCEPT: Why max_length?
        - AI models have a maximum input size (like a word limit)
        - For this model, it's 512 tokens (roughly 400-500 words)
        - Longer texts get cut off
        """
        
        # Handle empty or missing text
        if not text or pd.isna(text):
            # Return neutral/zero scores if no text
            return {
                'anger': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'joy': 0.0,
                'neutral': 1.0,
                'sadness': 0.0,
                'surprise': 0.0
            }
        
        # Truncate very long texts
        # CONCEPT: Tokenization
        # - Models have a maximum token limit (512 for this model)
        # - We truncate to ~450 tokens worth of text to be safe
        # - Rough estimate: 1 token ≈ 3-4 characters for English
        max_chars = 450 * 3  # Conservative: ~1350 characters
        if len(text) > max_chars:
            text = text[:max_chars]
        
        try:
            # Run the model on the text
            # This is where the magic happens!
            # Pass truncation=True to handle any remaining long texts
            result = self.emotion_classifier(text, truncation=True, max_length=512)[0]
            
            # Convert list of dicts to a single dict
            # CONCEPT: Data structure transformation
            # Model returns: [{'label': 'joy', 'score': 0.8}, {'label': 'sadness', 'score': 0.1}, ...]
            # We want: {'joy': 0.8, 'sadness': 0.1, ...}
            emotion_scores = {item['label']: item['score'] for item in result}
            
            return emotion_scores
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)[:100]}")
            # Return neutral if analysis fails
            return {
                'anger': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'joy': 0.0,
                'neutral': 1.0,
                'sadness': 0.0,
                'surprise': 0.0
            }
    
    def analyze_dataframe(self, df, text_column='lyric_cleaned', batch_size=8):
        """
        Analyze emotions for all songs in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with song lyrics
            text_column (str): Column name containing lyrics
            batch_size (int): Number of songs to process at once
            
        Returns:
            pd.DataFrame: Original DataFrame + emotion columns
            
        CONCEPT: Batch Processing
        - Instead of analyzing one song at a time, we do multiple together
        - This is more efficient (especially on GPU)
        - But we can't do ALL songs at once (would run out of memory)
        """
        print(f"\n=== Analyzing Emotions for {len(df)} Songs ===")
        print(f"Batch size: {batch_size}")
        
        # Create list to store all emotion scores
        all_emotions = []
        
        # Process in batches with progress bar
        # CONCEPT: tqdm gives us a progress bar so we know how long it'll take
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            # Get batch of lyrics
            batch = df[text_column].iloc[i:i+batch_size].tolist()
            
            # Analyze each lyric in the batch
            for text in batch:
                emotions = self.analyze_emotion(text)
                all_emotions.append(emotions)
        
        # Convert list of emotion dicts to DataFrame
        # CONCEPT: This creates new columns for each emotion
        emotion_df = pd.DataFrame(all_emotions)
        
        # Add emotion columns to original DataFrame
        # CONCEPT: pd.concat joins DataFrames side-by-side (like paste in Excel)
        df_with_emotions = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)
        
        print("\n✓ Emotion analysis complete!")
        print(f"Added emotion columns: {list(emotion_df.columns)}")
        
        return df_with_emotions
    
    def calculate_compound_features(self, df):
        """
        Create additional emotional features from basic emotions.
        
        Args:
            df (pd.DataFrame): DataFrame with emotion columns
            
        Returns:
            pd.DataFrame: DataFrame with additional features
            
        CONCEPT: Feature Engineering
        - We create new features by combining existing ones
        - This helps capture more nuanced emotions
        - Example: "bittersweet" = high sadness + medium joy
        """
        print("\n=== Creating Compound Emotional Features ===")
        
        # Valence: Overall positivity/negativity
        # CONCEPT: Valence is a psychology term for emotional positivity
        # Positive emotions (joy) increase it, negative emotions (sadness, anger) decrease it
        df['valence'] = (
            df['joy'] * 1.0 +           # Joy is very positive
            df['surprise'] * 0.5 +       # Surprise is somewhat positive
            df['neutral'] * 0.0 +        # Neutral is... neutral
            df['sadness'] * -1.0 +       # Sadness is negative
            df['anger'] * -1.0 +         # Anger is negative
            df['fear'] * -0.8 +          # Fear is pretty negative
            df['disgust'] * -0.8         # Disgust is pretty negative
        )
        
        # Arousal: Emotional intensity/energy
        # CONCEPT: Arousal measures how "activated" the emotion is
        # Both very happy and very angry are high arousal
        # Sadness and neutral are low arousal
        df['arousal'] = (
            df['anger'] * 1.0 +
            df['joy'] * 0.8 +
            df['surprise'] * 0.9 +
            df['fear'] * 0.7 +
            df['sadness'] * 0.3 +
            df['neutral'] * 0.0
        )
        
        # Dominance: Feeling of control vs powerlessness
        # CONCEPT: Do lyrics express empowerment or vulnerability?
        df['dominance'] = (
            df['anger'] * 0.8 +          # Anger can feel powerful
            df['joy'] * 0.6 +            # Joy feels good/in-control
            df['sadness'] * -0.7 +       # Sadness feels powerless
            df['fear'] * -0.9 +          # Fear is lack of control
            df['disgust'] * 0.2          # Disgust has some power (rejection)
        )
        
        # Normalize to 0-1 range
        # CONCEPT: Normalization makes values comparable
        # Different features have different ranges, normalization fixes that
        for col in ['valence', 'arousal', 'dominance']:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Emotional complexity: How mixed are the emotions?
        # CONCEPT: Some songs have clear single emotions, others are complex
        # High complexity = multiple strong emotions (bittersweet songs)
        # Low complexity = one dominant emotion
        emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        df['emotional_complexity'] = df[emotion_cols].std(axis=1)
        
        print("Added features:")
        print("  - valence (positive vs negative)")
        print("  - arousal (calm vs intense)")
        print("  - dominance (powerless vs empowered)")
        print("  - emotional_complexity (simple vs mixed emotions)")
        
        return df
    
    def get_dominant_emotion(self, df):
        """
        Identify the strongest emotion for each song.
        
        Args:
            df (pd.DataFrame): DataFrame with emotion columns
            
        Returns:
            pd.DataFrame: DataFrame with 'dominant_emotion' column
            
        CONCEPT: Dominant Emotion
        - While songs can have mixed emotions, one is usually strongest
        - This gives us a simple label for each song
        - Useful for quick filtering and understanding
        """
        emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # For each song, find which emotion has the highest score
        # CONCEPT: idxmax() returns the column name with maximum value
        df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
        
        # Count how many songs have each dominant emotion
        print("\n=== Dominant Emotion Distribution ===")
        print(df['dominant_emotion'].value_counts())
        
        return df


# Example usage and testing
if __name__ == "__main__":
    """
    Process all songs with sentiment analysis.
    """
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('data/processed/taylor_swift_processed.csv')
    
    print(f"\n=== Analyzing Emotions for {len(df)} Songs ===")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze ALL songs (not just 20!)
    # Use smaller batch size to avoid memory issues
    df = analyzer.analyze_dataframe(df, batch_size=4)
    df = analyzer.calculate_compound_features(df)
    df = analyzer.get_dominant_emotion(df)
    
    # Show results for a few songs
    print("\n=== Sample Results ===")
    display_cols = ['track_title', 'album_name', 'dominant_emotion', 'valence', 'arousal']
    print(df[display_cols].head(10))
    
    # Save FULL results to the expected location
    output_path = 'data/processed/taylor_swift_with_emotions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")
    print(f"✓ Processed {len(df)} songs with emotion analysis!")

