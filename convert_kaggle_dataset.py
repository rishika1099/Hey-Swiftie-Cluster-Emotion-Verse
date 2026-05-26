"""
KAGGLE DATASET CONVERTER
=========================
Converts the Kaggle Taylor Swift lyrics dataset (with individual .txt files) 
to the CSV format expected by the project pipeline.

Dataset: https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums/data

The Kaggle dataset structure:
- Individual .txt files for each song (in data/Albums/<album_name>/<song_name>.txt)
- CSV files listing albums and tracks (in data/Tabular/)

Expected output:
- Single CSV with columns: track_title, album_name, lyric
- All lyrics consolidated into one file for easy processing

Usage:
    python convert_kaggle_dataset.py
    
    # Or specify custom paths:
    python convert_kaggle_dataset.py --input archive.zip --output data/raw/taylor_swift_lyrics.csv
"""

import zipfile
import pandas as pd
import os
import re
import argparse
from pathlib import Path


def canonical_album_key(name: str) -> str:
    """
    Normalize an album name (or a cover-art filename stem) into a single
    canonical key so we can match them up despite the dataset's two
    different escaping schemes:

      - Folder name:    "1989_TaylorsVersion_"        (apostrophe stripped)
      - Cover filename: "1989_Taylor_sVersion_.jpg"   (apostrophe → underscore)

    Both should canonicalize to "1989taylorsversion".
    """
    s = name.lower()
    # Drop anything that's not a letter or digit. This kills underscores,
    # spaces, apostrophes, parens, hyphens — everything.
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def extract_cover_art(zip_path, output_dir="cover_art"):
    """
    Extract album cover images from data/Cover_Art/ in the Kaggle zip
    and save them into cover_art/ with canonical filenames so the
    backend can match them by album_name.

    Args:
        zip_path (str): Path to the Kaggle archive.zip
        output_dir (str): Where to write the cover images
    """
    print("\n" + "=" * 60)
    print("EXTRACTING ALBUM COVER ART".center(60))
    print("=" * 60 + "\n")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        cover_files = [
            f for f in z.namelist()
            if f.startswith("data/Cover_Art/")
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        print(f"🖼️  Found {len(cover_files)} cover images\n")

        for cover in cover_files:
            stem = Path(cover).stem            # e.g. "1989_Taylor_sVersion_"
            ext = Path(cover).suffix.lower()   # e.g. ".jpg"
            key = canonical_album_key(stem)     # e.g. "1989taylorsversion"
            if not key:
                continue

            dest = out / f"{key}{ext}"
            with z.open(cover) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            saved += 1

    print(f"✓ Saved {saved} covers to: {out.resolve()}")
    return saved


def convert_kaggle_dataset(zip_path, output_path):
    """
    Convert Kaggle dataset to expected CSV format.
    
    Args:
        zip_path (str): Path to the downloaded archive.zip from Kaggle
        output_path (str): Where to save taylor_swift_lyrics.csv
        
    Returns:
        pd.DataFrame: The converted dataset
    """
    print("\n" + "="*60)
    print("CONVERTING KAGGLE DATASET".center(60))
    print("="*60 + "\n")
    
    all_songs = []
    skipped_files = []
    
    # Open the zip file
    print(f"📦 Reading dataset from: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get all text files (these contain the lyrics)
        txt_files = [f for f in z.namelist() if f.endswith('.txt') and 'data/Albums/' in f]
        
        print(f"📝 Found {len(txt_files)} lyric files\n")
        print("Processing...")
        
        for txt_file in txt_files:
            # Extract album and song name from path
            # Example path: data/Albums/Lover/Lover.txt
            parts = txt_file.split('/')
            
            if len(parts) >= 4:
                album_name = parts[2]  # "Lover"
                song_file = parts[3]    # "Lover.txt"
                
                # Clean up song name
                # Remove .txt extension and replace underscores with spaces
                song_name = song_file.replace('.txt', '').replace('_', ' ')
                
                # Skip special files that aren't actual songs
                skip_keywords = [
                    'liner', 'prologue', 'booklet', 'poem', 'magazine',
                    'notes', 'credits', 'thank', 'dedication'
                ]
                
                if any(keyword in song_name.lower() for keyword in skip_keywords):
                    skipped_files.append(song_name)
                    continue
                
                # Read the lyrics
                try:
                    with z.open(txt_file) as f:
                        lyrics = f.read().decode('utf-8', errors='ignore')
                    
                    # Only include if has substantial lyrics (more than 50 characters)
                    if len(lyrics) > 50:
                        all_songs.append({
                            'track_title': song_name,
                            'album_name': album_name,
                            'lyric': lyrics
                        })
                        
                except Exception as e:
                    print(f"⚠️  Error reading {txt_file}: {e}")
                    skipped_files.append(song_name)
    
    # Create DataFrame
    df = pd.DataFrame(all_songs)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY".center(60))
    print("="*60 + "\n")
    
    print(f"✓ Successfully processed: {len(df)} songs")
    print(f"✓ From {df['album_name'].nunique()} unique albums")
    print(f"✓ Skipped {len(skipped_files)} non-song files (liner notes, etc.)")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Show statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS".center(60))
    print("="*60 + "\n")
    
    print(f"Total songs: {len(df)}")
    print(f"Total albums: {df['album_name'].nunique()}")
    
    # Word count statistics
    df['word_count'] = df['lyric'].str.split().str.len()
    print(f"\nLyrics statistics:")
    print(f"  Average words per song: {df['word_count'].mean():.0f}")
    print(f"  Shortest song: {df['word_count'].min()} words")
    print(f"  Longest song: {df['word_count'].max()} words")
    
    # Top albums by song count
    print("\n📊 Top 10 albums by song count:")
    album_counts = df['album_name'].value_counts().head(10)
    for i, (album, count) in enumerate(album_counts.items(), 1):
        print(f"  {i:2d}. {album[:45]:<45} {count:3d} songs")
    
    # Show sample of converted data
    print("\n" + "="*60)
    print("SAMPLE DATA".center(60))
    print("="*60 + "\n")
    
    sample_cols = ['track_title', 'album_name']
    print(df[sample_cols].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE".center(60))
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Place the CSV in: data/raw/taylor_swift_lyrics.csv")
    print("  2. Run: python setup_and_run.py --all")
    print()
    
    return df


def main():
    """
    Main function to handle command-line arguments and run conversion.
    """
    parser = argparse.ArgumentParser(
        description='Convert Kaggle Taylor Swift dataset to project CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (with default paths):
  python convert_kaggle_dataset.py
  
  # Custom input and output paths:
  python convert_kaggle_dataset.py --input ~/Downloads/archive.zip --output data/raw/lyrics.csv
  
  # Show what would be converted without saving:
  python convert_kaggle_dataset.py --dry-run
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='archive.zip',
        help='Path to the Kaggle archive.zip file (default: archive.zip)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/raw/taylor_swift_lyrics.csv',
        help='Output CSV path (default: data/raw/taylor_swift_lyrics.csv)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without saving the file'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"\n❌ Error: Input file not found: {args.input}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums/data")
        print("\nAnd save it as 'archive.zip' in the current directory")
        print("Or specify the path with: --input path/to/archive.zip")
        return
    
    # Run conversion
    try:
        df = convert_kaggle_dataset(args.input, args.output)

        if args.dry_run:
            print("🔍 DRY RUN - File was NOT saved")
            print(f"   (Would have saved to: {args.output})")
        else:
            # Also pull out the album cover images from data/Cover_Art/
            extract_cover_art(args.input, output_dir="cover_art")
            print(f"✅ Success! Dataset ready at: {args.output}")
            
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf the error persists, please check:")
        print("  - The zip file is not corrupted")
        print("  - You have write permissions to the output directory")
        print("  - You have enough disk space")


if __name__ == "__main__":
    main()
