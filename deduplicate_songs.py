"""
SONG DEDUPLICATION SCRIPT
=========================
Removes duplicate song versions and keeps only one album per song.
"""

import pandas as pd
from pathlib import Path

def deduplicate_songs(input_file, output_file):
    """
    Remove duplicate songs and keep only one version.
    
    Strategy:
    1. Group by track_title (song name)
    2. For each song, keep the version from the MAIN album (not deluxe/vault)
    3. If all versions are special, keep the first one
    """
    
    print("="*60)
    print("SONG DEDUPLICATION")
    print("="*60)
    
    # Read the data
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df)} total songs")
    print(f"  Unique song titles: {df['track_title'].nunique()}")
    print(f"  Albums: {df['album_name'].nunique()}")
    
    # Define priority for album selection
    # Prefer standard editions over special editions
    def get_album_priority(album_name):
        """Lower number = higher priority."""
        album_lower = album_name.lower()
        
        # Highest priority: standard albums
        if 'deluxe' not in album_lower and 'vault' not in album_lower and 'version' not in album_lower:
            return 0
        # Medium: deluxe editions
        elif 'deluxe' in album_lower:
            return 1
        # Lower: vault editions
        elif 'vault' in album_lower:
            return 2
        # Lowest: other versions
        else:
            return 3
    
    # Add priority column
    df['album_priority'] = df['album_name'].apply(get_album_priority)
    
    # Sort by track title, then by priority (lower = better)
    df_sorted = df.sort_values(['track_title', 'album_priority'])
    
    # Keep first occurrence of each song (which will be highest priority)
    df_dedup = df_sorted.drop_duplicates(subset='track_title', keep='first')
    
    # Remove the temporary priority column
    df_dedup = df_dedup.drop('album_priority', axis=1)
    
    # Sort by album and track
    df_dedup = df_dedup.sort_values(['album_name', 'track_title'])
    
    # Reset index
    df_dedup = df_dedup.reset_index(drop=True)
    
    print(f"\n✓ After deduplication:")
    print(f"  Total songs: {len(df_dedup)} (removed {len(df) - len(df_dedup)} duplicates)")
    print(f"  Unique song titles: {df_dedup['track_title'].nunique()}")
    print(f"  Albums remaining: {df_dedup['album_name'].nunique()}")
    
    # Show some examples of what was kept
    print(f"\n📊 Sample of deduplicated songs:")
    sample = df_dedup[['track_title', 'album_name']].head(10)
    for idx, row in sample.iterrows():
        print(f"  - {row['track_title']:40} ({row['album_name']})")
    
    # Save
    print(f"\n💾 Saving to: {output_file}")
    df_dedup.to_csv(output_file, index=False)
    
    print(f"\n✓ Deduplication complete!")
    print("="*60)
    
    return df_dedup


def analyze_duplicates(df):
    """Show which songs had multiple versions."""
    
    print("\n" + "="*60)
    print("DUPLICATE ANALYSIS")
    print("="*60)
    
    # Find songs with multiple versions
    song_counts = df['track_title'].value_counts()
    duplicates = song_counts[song_counts > 1]
    
    if len(duplicates) == 0:
        print("\n✓ No duplicates found!")
        return
    
    print(f"\nSongs with multiple versions: {len(duplicates)}")
    print(f"\nTop 10 most duplicated songs:")
    
    for song, count in duplicates.head(10).items():
        print(f"\n  {song} ({count} versions):")
        versions = df[df['track_title'] == song][['album_name', 'dominant_emotion']]
        for idx, row in versions.iterrows():
            print(f"    - {row['album_name']} (Vibe: {row['dominant_emotion']})")


if __name__ == "__main__":
    # File paths
    input_file = 'data/processed/taylor_swift_clustered.csv'
    output_file = 'data/processed/taylor_swift_clustered_dedup.csv'
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        print("   Please run the clustering step first!")
        exit(1)
    
    # Analyze duplicates before deduplication
    print("\n📊 BEFORE DEDUPLICATION:")
    df_original = pd.read_csv(input_file)
    analyze_duplicates(df_original)
    
    # Deduplicate
    df_dedup = deduplicate_songs(input_file, output_file)
    
    # Analyze after
    print("\n📊 AFTER DEDUPLICATION:")
    analyze_duplicates(df_dedup)
    
    print("\n✅ Done! Updated file saved to:")
    print(f"   {output_file}")
    print("\n💡 To use the deduplicated version:")
    print("   Update app.py line 36:")
    print("   diary_classifier = DiaryClassifier('data/processed/taylor_swift_clustered_dedup.csv')")
