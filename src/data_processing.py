"""
DATA PROCESSING MODULE
======================
This module handles loading, cleaning, and preprocessing the Taylor Swift lyrics dataset.

KEY CONCEPTS:
- Data cleaning: Removing duplicates, handling missing values
- Text preprocessing: Preparing lyrics for analysis
- Feature engineering: Creating useful columns from raw data
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

class DataProcessor:
    """
    A class to handle all data processing operations.
    
    CONCEPT: Object-Oriented Programming
    - We use a class to organize related functions together
    - This makes code more maintainable and reusable
    """
    
    def __init__(self, data_path):
        """
        Initialize the processor with the path to the dataset.
        
        Args:
            data_path (str): Path to the CSV file from Kaggle
            
        CONCEPT: Constructor (__init__)
        - This is called when you create a new DataProcessor object
        - It sets up the initial state of the object
        """
        self.data_path = Path(data_path)
        self.df = None  # Will store our DataFrame
        
    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded data
            
        CONCEPT: Pandas DataFrame
        - Think of it like an Excel spreadsheet in Python
        - Rows = individual songs, Columns = attributes (title, lyrics, album, etc.)
        """
        print("Loading data from:", self.data_path)
        
        # Read CSV file into DataFrame
        # encoding='utf-8' handles special characters in lyrics
        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        
        print(f"Loaded {len(self.df)} songs")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def clean_data(self):
        """
        Clean the dataset by removing duplicates and handling missing values.
        
        CONCEPT: Data Cleaning
        - Real-world data is messy! We need to clean it before analysis
        - Common issues: duplicates, missing values, inconsistent formatting
        """
        print("\n=== Starting Data Cleaning ===")
        initial_rows = len(self.df)
        
        # 1. Remove duplicate songs (same title + album)
        # CONCEPT: Duplicates can skew our analysis, making some songs over-represented
        self.df = self.df.drop_duplicates(subset=['track_title', 'album_name'], keep='first')
        duplicates_removed = initial_rows - len(self.df)
        print(f"Removed {duplicates_removed} duplicate songs")
        
        # 2. Handle missing lyrics
        # CONCEPT: Songs without lyrics can't be analyzed, so we remove them
        missing_lyrics = self.df['lyric'].isna().sum()
        self.df = self.df.dropna(subset=['lyric'])
        print(f"Removed {missing_lyrics} songs with missing lyrics")
        
        # 3. Remove instrumental tracks (very short or no lyrics)
        # CONCEPT: Instrumental tracks have no emotional content to analyze
        self.df['lyric_length'] = self.df['lyric'].str.len()
        instrumentals = (self.df['lyric_length'] < 50).sum()
        self.df = self.df[self.df['lyric_length'] >= 50]
        print(f"Removed {instrumentals} instrumental/very short tracks")
        
        # 4. Standardize column names (lowercase, no spaces)
        # CONCEPT: Consistent naming makes code easier to write and less error-prone
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        
        print(f"\nFinal dataset: {len(self.df)} songs")
        return self.df
    
    def preprocess_lyrics(self):
        """
        Preprocess lyrics text for analysis.
        
        CONCEPT: Text Preprocessing
        - Raw text needs cleaning before we can analyze it
        - We want to keep meaningful content while removing noise
        """
        print("\n=== Preprocessing Lyrics ===")
        
        def clean_lyric_text(text):
            """
            Clean individual lyric text.
            
            Steps:
            1. Remove special annotations like [Verse], [Chorus]
            2. Remove extra whitespace
            3. Keep punctuation (it carries emotional meaning!)
            """
            if pd.isna(text):
                return ""
            
            # Remove section markers like [Verse 1], [Chorus], etc.
            # CONCEPT: Regex (Regular Expressions) - pattern matching for text
            # \[.*?\] means: match anything between square brackets
            text = re.sub(r'\[.*?\]', '', text)
            
            # Remove multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
        
        # Apply cleaning function to all lyrics
        # CONCEPT: .apply() runs a function on every row in the DataFrame
        self.df['lyric_cleaned'] = self.df['lyric'].apply(clean_lyric_text)
        
        # Count words in each song (useful feature!)
        # CONCEPT: Word count can indicate song complexity/depth
        self.df['word_count'] = self.df['lyric_cleaned'].str.split().str.len()
        
        print(f"Average words per song: {self.df['word_count'].mean():.0f}")
        print(f"Range: {self.df['word_count'].min()} to {self.df['word_count'].max()}")
        
        return self.df
    
    def create_album_metadata(self):
        """
        Create useful metadata about albums for theming.
        
        CONCEPT: Feature Engineering
        - Creating new useful information from existing data
        - This will help us map songs to album themes later
        """
        print("\n=== Creating Album Metadata ===")
        
        # Map albums to their release era
        # CONCEPT: Dictionary mapping - key:value pairs for quick lookups
        # This is based on Taylor Swift's actual album releases
        album_to_era = {
            'Taylor Swift': 'debut',
            'Fearless': 'fearless',
            'Speak Now': 'speak_now',
            'Red': 'red',
            '1989': '1989',
            'reputation': 'reputation',
            'Lover': 'lover',
            'folklore': 'folklore',
            'evermore': 'evermore',
            'Midnights': 'midnights',
            'THE TORTURED POETS DEPARTMENT': 'ttpd'
        }
        
        # Add era column
        # CONCEPT: .map() replaces values based on a dictionary
        self.df['era'] = self.df['album_name'].map(album_to_era)
        
        # Handle any albums not in our mapping (deluxe versions, etc.)
        self.df['era'] = self.df['era'].fillna('other')
        
        # Count songs per album
        album_counts = self.df['album_name'].value_counts()
        print(f"\nSongs per album:")
        print(album_counts.head(10))
        
        return self.df
    
    def save_processed_data(self, output_path):
        """
        Save the cleaned and processed data.
        
        Args:
            output_path (str): Where to save the processed CSV
            
        CONCEPT: Saving intermediate results
        - We save processed data so we don't have to re-run cleaning every time
        - This makes development faster
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to: {output_path}")
        
    def get_summary_statistics(self):
        """
        Get a summary of the processed dataset.
        
        CONCEPT: Exploratory Data Analysis (EDA)
        - Understanding your data before building models
        - Helps catch issues and understand patterns
        """
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        print(f"\nTotal Songs: {len(self.df)}")
        print(f"Total Albums: {self.df['album_name'].nunique()}")
        print(f"Total Eras: {self.df['era'].nunique()}")
        
        print("\n--- Lyric Statistics ---")
        print(f"Average words per song: {self.df['word_count'].mean():.0f}")
        print(f"Median words per song: {self.df['word_count'].median():.0f}")
        print(f"Shortest song: {self.df['word_count'].min()} words")
        print(f"Longest song: {self.df['word_count'].max()} words")
        
        print("\n--- Era Distribution ---")
        print(self.df['era'].value_counts())


# Example usage and testing
if __name__ == "__main__":
    """
    CONCEPT: __main__ block
    - This code only runs when you execute this file directly
    - Allows us to test the module independently
    """
    
    # Initialize processor
    processor = DataProcessor('data/raw/taylor_swift_lyrics.csv')
    
    # Run the full pipeline
    processor.load_data()
    processor.clean_data()
    processor.preprocess_lyrics()
    processor.create_album_metadata()
    
    # Get summary
    processor.get_summary_statistics()
    
    # Save processed data
    processor.save_processed_data('data/processed/taylor_swift_processed.csv')
    
    print("\n✓ Data processing complete!")
