"""
Build a vector index of Taylor Swift stanzas for retrieval-augmented
verse generation.

Pipeline
--------
1. Read the raw lyrics CSV and the clustered+enriched CSV.
2. Inner-join on (track_title, album_name) so each lyric row carries
   its emotion cluster id. This also dedupes to ~one row per song,
   since the enriched CSV is dedup'd.
3. Split each song's lyrics into stanzas (`\\n\\n`-separated), strip
   section headers like "[Chorus]" or "[Verse 1]", drop very short ones.
4. Embed every stanza with sentence-transformers/all-MiniLM-L6-v2.
   (Small, fast, runs on CPU. 384 dim, ~80MB model.)
5. Save a FAISS IndexFlatIP (inner product over normalized vectors)
   plus a JSON sidecar with each chunk's metadata.

At request time the backend uses `cluster_id` to post-filter retrieved
chunks so they share the diary's emotional space.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_LYRICS = ROOT / "data" / "raw" / "taylor_swift_lyrics.csv"
ENRICHED = ROOT / "data" / "processed" / "taylor_swift_clustered_enriched.csv"
DEDUP_FALLBACK = ROOT / "data" / "processed" / "taylor_swift_clustered_dedup.csv"

INDEX_OUT = ROOT / "data" / "processed" / "lyric_index.faiss"
CHUNKS_OUT = ROOT / "data" / "processed" / "lyric_chunks.json"

MIN_STANZA_LEN = 24       # chars
MIN_STANZA_LINES = 2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_stanzas(lyric: str) -> list[str]:
    """Break a lyric blob into clean stanzas suitable for retrieval."""
    if not isinstance(lyric, str):
        return []

    # Drop section labels like "[Chorus]", "[Verse 1]", "[Bridge]"
    lyric = re.sub(r"\[[^\]]*\]", "", lyric)
    # Drop "(repeat 2x)" type annotations
    lyric = re.sub(r"\((?:repeat|x\d+|\d+x)[^)]*\)", "", lyric, flags=re.IGNORECASE)

    pieces = re.split(r"\n\s*\n+", lyric)
    out = []
    for p in pieces:
        cleaned = p.strip()
        if len(cleaned) < MIN_STANZA_LEN:
            continue
        if cleaned.count("\n") + 1 < MIN_STANZA_LINES:
            continue
        # Collapse extra whitespace within lines but preserve line breaks
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
        out.append(cleaned)
    return out


def main() -> None:
    if not RAW_LYRICS.exists():
        raise SystemExit(f"❌ Missing {RAW_LYRICS}. Run convert_kaggle_dataset.py first.")
    src = ENRICHED if ENRICHED.exists() else DEDUP_FALLBACK
    if not src.exists():
        raise SystemExit(f"❌ Missing {src}. Run the clustering pipeline first.")

    print(f"📥 Lyrics: {RAW_LYRICS.name}")
    print(f"📥 Clusters: {src.name}")

    lyrics_df = pd.read_csv(RAW_LYRICS)
    clusters_df = pd.read_csv(src)

    merged = lyrics_df.merge(
        clusters_df[["track_title", "album_name", "cluster"]],
        on=["track_title", "album_name"],
        how="inner",
    )
    print(f"🔗 Joined: {len(merged)} song-version rows with cluster ids")

    # Build the chunk list
    chunks = []
    for _, row in merged.iterrows():
        for stanza in split_stanzas(row["lyric"]):
            chunks.append(
                {
                    "text": stanza,
                    "track_title": row["track_title"],
                    "album_name": row["album_name"],
                    "cluster": int(row["cluster"]),
                }
            )
    print(f"📜 Total stanzas: {len(chunks)}")

    if not chunks:
        raise SystemExit("❌ No stanzas extracted — check the lyrics CSV format.")

    # Embed
    print(f"🤖 Loading embedder: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    print("🔢 Embedding…")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    print(f"   embeddings shape: {embeddings.shape}")

    # Build FAISS index (inner-product over unit vectors == cosine sim)
    print("🗄️  Building FAISS IndexFlatIP")
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_OUT))
    with open(CHUNKS_OUT, "w") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"✅ Wrote {INDEX_OUT}")
    print(f"✅ Wrote {CHUNKS_OUT}")
    print()
    print("Cluster distribution:")
    counts = pd.Series([c["cluster"] for c in chunks]).value_counts().sort_index()
    for cid, n in counts.items():
        print(f"   cluster {cid}: {n} stanzas")


if __name__ == "__main__":
    main()
