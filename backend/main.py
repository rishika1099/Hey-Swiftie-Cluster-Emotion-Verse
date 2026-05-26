"""
FastAPI backend for the Taylor Swift Letter Generator.

Exposes a single POST /api/generate endpoint that the React frontend
calls with a diary entry and gets back:
  - poem (Taylor-style verse)
  - theme (album aesthetic name + palette)
  - emotions (per-emotion scores)
  - cluster (id + label)
  - songs (top recommended songs with similarity)
"""

import base64
import os
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make ../src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sentiment_analyzer import SentimentAnalyzer
from src.diary_classifier import DiaryClassifier
from src.theme_manager import ThemeManager
from src.letter_generator_openai import LetterGenerator


CLUSTERED_ENRICHED_CSV = ROOT / "data" / "processed" / "taylor_swift_clustered_enriched.csv"
CLUSTERED_DEDUP_CSV = ROOT / "data" / "processed" / "taylor_swift_clustered_dedup.csv"
CLUSTERED_CSV_FALLBACK = ROOT / "data" / "processed" / "taylor_swift_clustered.csv"
COVER_DIR = ROOT / "cover_art"

LYRIC_INDEX_PATH = ROOT / "data" / "processed" / "lyric_index.faiss"
LYRIC_CHUNKS_PATH = ROOT / "data" / "processed" / "lyric_chunks.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_clustered_csv() -> Path:
    """Prefer the Spotify-enriched CSV, then dedup, then the raw clustered one."""
    for p in (CLUSTERED_ENRICHED_CSV, CLUSTERED_DEDUP_CSV, CLUSTERED_CSV_FALLBACK):
        if p.exists():
            return p
    return CLUSTERED_CSV_FALLBACK


# Spotify metadata lookup, populated at startup. Keys are
# (track_title, album_name) tuples — the same identifiers
# DiaryClassifier returns in its recommendations.
SPOTIFY_LOOKUP: dict[tuple[str, str], dict] = {}


app = FastAPI(title="Dear Diary, Love Taylor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


emotion_analyzer: SentimentAnalyzer | None = None
diary_classifier: DiaryClassifier | None = None
theme_manager: ThemeManager | None = None
letter_generator: LetterGenerator | None = None

# Vector RAG state
lyric_index = None       # faiss.IndexFlatIP
lyric_chunks: list[dict] = []
embedder = None          # SentenceTransformer


@app.on_event("startup")
def load_models() -> None:
    global emotion_analyzer, diary_classifier, theme_manager, letter_generator

    csv_path = _pick_clustered_csv()
    if not csv_path.exists():
        print(
            f"⚠️  {csv_path} missing. Run the data pipeline before /api/generate works:\n"
            "    python convert_kaggle_dataset.py\n"
            "    python src/data_processing.py\n"
            "    python src/sentiment_analyzer.py\n"
            "    python src/clustering.py\n"
            "    python deduplicate_songs.py    # optional but recommended"
        )
        return

    print(f"✓ Using clustered dataset: {csv_path.name}")
    emotion_analyzer = SentimentAnalyzer()
    diary_classifier = DiaryClassifier(str(csv_path))

    # Build the Spotify lookup table from the enriched CSV if available.
    if csv_path == CLUSTERED_ENRICHED_CSV:
        import pandas as pd
        enriched = pd.read_csv(csv_path)
        if "spotify_id" in enriched.columns:
            hits = 0
            for _, row in enriched.iterrows():
                tid = row.get("spotify_id")
                if not isinstance(tid, str) or not tid:
                    continue
                SPOTIFY_LOOKUP[(row["track_title"], row["album_name"])] = {
                    "spotify_name": row.get("spotify_name", ""),
                    "spotify_album": row.get("spotify_album", ""),
                    "spotify_id": tid,
                    "spotify_uri": row.get("spotify_uri", ""),
                }
                hits += 1
            print(f"✓ Loaded Spotify metadata for {hits} / {len(enriched)} songs")
    theme_manager = ThemeManager()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set — letter generation will use fallback templates.")
    letter_generator = LetterGenerator("gpt-4o-mini", api_key=api_key or "missing")

    # Vector RAG: load FAISS index + chunk metadata + sentence-transformer model.
    # Optional — if the artifacts are missing we just generate without retrieval.
    global lyric_index, lyric_chunks, embedder
    if LYRIC_INDEX_PATH.exists() and LYRIC_CHUNKS_PATH.exists():
        import json
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"✓ Loading lyric RAG index from {LYRIC_INDEX_PATH.name}")
        lyric_index = faiss.read_index(str(LYRIC_INDEX_PATH))
        with open(LYRIC_CHUNKS_PATH) as f:
            lyric_chunks = json.load(f)
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"✓ Lyric RAG ready ({len(lyric_chunks)} stanzas, dim={lyric_index.d})")
    else:
        print("ℹ️  No lyric_index.faiss — verse generation will skip RAG.")


class GenerateRequest(BaseModel):
    diary: str = Field(..., min_length=10, max_length=4000)


# Words that should stay lowercase in a Title Case rendering.
_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with",
}

# Phrases stripped from track/album names before display — re-recording
# markers, vault tags, edition labels, etc. Order matters (longer first).
_STRIP_PATTERNS = [
    r"taylor'?s?\s*version",
    r"from\s+the\s+vault",
    r"the\s+from\s+the\s+vault\s+chapter",
    r"record\s+store\s+day\s+exclusive",
    r"digital\s+deluxe",
    r"platinum\s+edition",
    r"deluxe\s+(?:edition|version)?",
    r"the\s+(?:long\s+pond|albatross|black\s+dog|bolter)\s+(?:studio\s+sessions|edition)?",
    r"japanese\s+edition",
    r"target\s+exclusive",
    r"apple\s+music\s+exclusive",
    r"first\s+draft\s+phone\s+memo",
    r"physical\s+version",
    r"tangerine\s+edition",
    r"webstore\s+deluxe",
    r"heart\s+shaped\s+vinyl",
    r"live\s+from\s+[a-z\s]+",
    r"3am\s+edition",
    r"the\s+til{1,2}\s+dawn\s+edition",
    r"the\s+late\s+night\s+edition",
    r"world\s+tour\s+live",
    # Chapter-edition descriptors. Real Taylor songs never contain "Chapter",
    # so the pattern below strips any "<connector-prefixed phrase> Chapter".
    # Requiring a connector word (the/from/of/with/etc.) prevents the
    # pattern from eating an album name immediately before "Chapter".
    r"(?:[A-Za-z'\-]+\s+)*\b(?:the|from|of|with|to|a|an|in|on|by|but|and|what|i)\b[A-Za-z'\-\s]*chapter\b",
    # Final fallback: a lone "Chapter" left behind.
    r"\bchapter\b",
]


def _segment_lowercase(word: str) -> str:
    """Split an all-lowercase glued word like 'thelastgreatamericandynasty'
    into separate words. Uses wordninja if available."""
    try:
        import wordninja
        pieces = wordninja.split(word)
        # Reject suspicious splits — e.g. "midnights" → ["midnight", "s"].
        # Single-character pieces other than 'a' / 'i' are almost always
        # plural- or possessive-tail false positives.
        if any(len(p) == 1 and p.lower() not in ("a", "i") for p in pieces):
            return word
        # Only accept if pieces actually re-form the word (no dropped chars).
        if pieces and sum(len(p) for p in pieces) >= len(word) * 0.9:
            return " ".join(pieces)
    except ImportError:
        pass
    return word


def _clean_name(raw: str) -> str:
    """Turn a raw dataset name like 'AllYouHadtoDoWasStay_TaylorsVersion_'
    into a human-readable 'All You Had to Do Was Stay'."""
    s = raw.replace("**", "").replace("*", "").replace("_", " ")

    # Split classic camelCase: 'AllYou' → 'All You'
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # Insert space at digit boundaries: '1989Taylors' → '1989 Taylors'
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)

    # Strip version/edition markers (case-insensitive)
    for pat in _STRIP_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # Run wordninja on any remaining all-lowercase glued run (e.g.
    # "thelastgreatamericandynasty" or post-camel-split fragments like
    # "Niceto" which are mixed case but still need re-segmenting).
    pieces = []
    for w in s.split():
        # Strip residual punctuation for the lookup, keep an apostrophe.
        token = re.sub(r"[^a-zA-Z']", "", w)
        if token and token.lower() == token and len(token) > 4:
            pieces.append(_segment_lowercase(token))
        elif token and not any(c.isupper() for c in token[1:]):
            # Mostly-lowercase like "Niceto" or "Havea" — also segment.
            seg = _segment_lowercase(token.lower())
            # Preserve the original capitalization on the first piece.
            if token[0].isupper() and seg:
                seg = seg[0].upper() + seg[1:]
            pieces.append(seg)
        else:
            pieces.append(w)
    s = " ".join(pieces)

    # Smart Title Case: capitalize everything except minor connectors
    # (unless they're the first or last word).
    words = s.split()
    cased = []
    for i, w in enumerate(words):
        low = w.lower()
        if 0 < i < len(words) - 1 and low in _MINOR_WORDS:
            cased.append(low)
        else:
            cased.append(w[:1].upper() + w[1:] if w else w)
    return " ".join(cased)


# Patterns that mark a re-recording, vault track, deluxe edit, or length variant.
# Stripped (case-insensitive) when computing a canonical title for dedup.
_VERSION_MARKERS = re.compile(
    r"""
    \s*
    (?:
        \(\s*taylor'?s?\s+version\s*\) |
        \[\s*taylor'?s?\s+version\s*\] |
        \(\s*from\s+the\s+vault[^)]*\) |
        \[\s*from\s+the\s+vault[^]]*\] |
        \(\s*\d+\s*minute\s+version\s*\) |
        \(\s*deluxe[^)]*\) |
        \(\s*acoustic\s*\) |
        \(\s*piano\s+version\s*\) |
        \(\s*pop\s+version\s*\) |
        \(\s*demo\s*\) |
        \(\s*remix[^)]*\) |
        \(\s*live[^)]*\) |
        \(\s*radio\s+edit\s*\) |
        \(\s*voice\s+memo\s*\) |
        \(\s*extended[^)]*\)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _canonical_title(raw: str) -> str:
    """Normalize a track title so versions of the same song collide."""
    s = _clean_name(raw).lower()
    # Strip every version marker, possibly multiple in a row
    prev = None
    while prev != s:
        prev = s
        s = _VERSION_MARKERS.sub("", s)
    # Drop residual punctuation and collapse whitespace
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _canonical_album_key(name: str) -> str:
    """Mirror of convert_kaggle_dataset.canonical_album_key — keep in sync."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_cover(album_name: str) -> str | None:
    """Return a data: URL for the album cover, or None if missing.

    Tries the full album name first, then progressively drops trailing
    qualifiers (deluxe / vault / chapter / target exclusive / …) so a
    track recommended from a deluxe edition can still match the base
    cover when only the base one was extracted.
    """
    if not COVER_DIR.exists():
        return None

    # Build a list of candidates: full name, then progressively trim
    # parenthetical/bracketed qualifiers from the right.
    candidates = [album_name]
    trimmed = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*$", "", album_name).strip()
    while trimmed and trimmed != candidates[-1]:
        candidates.append(trimmed)
        trimmed = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*$", "", trimmed).strip()

    for name in candidates:
        key = _canonical_album_key(name)
        if not key:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = COVER_DIR / f"{key}{ext}"
            if p.exists():
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
                data = base64.b64encode(p.read_bytes()).decode()
                return f"data:{mime};base64,{data}"
    return None


def _retrieve_stanzas(
    diary_text: str,
    cluster_id: int,
    k: int = 6,
    pool: int = 80,
) -> list[dict]:
    """Pull cluster-filtered Taylor stanzas most similar to the diary.

    The FAISS index is unfiltered (one big haystack), so we over-retrieve
    `pool` candidates by raw similarity, then keep the first `k` whose
    cluster matches the diary's matched cluster — and dedup by song so we
    don't return the same song twice with two stanzas.
    """
    if lyric_index is None or embedder is None or not lyric_chunks:
        return []
    import numpy as np

    query = embedder.encode([diary_text], normalize_embeddings=True).astype("float32")
    scores, idxs = lyric_index.search(query, pool)

    out, seen_songs = [], set()
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        chunk = lyric_chunks[i]
        if chunk["cluster"] != cluster_id:
            continue
        title = chunk["track_title"]
        if title in seen_songs:
            continue
        seen_songs.add(title)
        out.append({**chunk, "score": float(score)})
        if len(out) >= k:
            break
    return out


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "models_loaded": all(
            x is not None for x in (emotion_analyzer, diary_classifier, theme_manager, letter_generator)
        ),
        "clustered_csv_exists": _pick_clustered_csv().exists(),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    if any(x is None for x in (emotion_analyzer, diary_classifier, theme_manager, letter_generator)):
        raise HTTPException(
            status_code=503,
            detail="Backend not ready. Run the data pipeline and set OPENAI_API_KEY, then restart.",
        )

    classification = diary_classifier.classify_diary(req.diary, emotion_analyzer)
    cluster_stats = diary_classifier.get_cluster_stats(classification["cluster_id"])
    theme = theme_manager.get_theme_for_cluster(cluster_stats)

    # Retrieve Taylor's own stanzas matching the diary's emotional space,
    # then feed them to GPT as style anchors (vector RAG).
    examples = _retrieve_stanzas(req.diary, classification["cluster_id"], k=6)
    poem = letter_generator.generate_letter(
        req.diary, classification, theme, temperature=0.9, examples=examples
    )

    # Dedup recommended songs by canonical title (e.g. "All Too Well" and
    # "All Too Well (10 Minute Version) (From The Vault)" collapse to one).
    # Keep the version with the highest similarity score.
    seen: dict[str, dict] = {}
    for s in classification["recommended_songs"]:
        canon = _canonical_title(s["track_title"])
        if not canon:
            continue
        sim = float(s["similarity"])
        if canon in seen and seen[canon]["similarity"] >= sim:
            continue

        spot = SPOTIFY_LOOKUP.get((s["track_title"], s["album_name"]))

        # Prefer the official Spotify name/album when we matched against the
        # Spotify catalogue; otherwise fall back to our regex-cleaned version
        # of the raw dataset name.
        display_title = spot["spotify_name"] if spot else _clean_name(s["track_title"])
        display_album = spot["spotify_album"] if spot else _clean_name(s["album_name"])

        seen[canon] = {
            "track_title": display_title,
            "album_name": display_album,
            "similarity": sim,
            "dominant_emotion": s.get("dominant_emotion"),
            "cover": _find_cover(s["album_name"]),
            "spotify_id": spot["spotify_id"] if spot else None,
        }

    songs = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)[:8]

    return {
        "poem": poem,
        "theme": {
            "name": theme["name"],
            "primary": theme["primary_color"],
            "secondary": theme["secondary_color"],
            "accent": theme["accent_color"],
            "background": theme["background"],
            "vibe": theme["vibe"],
        },
        "cluster": {
            "id": classification["cluster_id"],
            "label": classification["cluster_label"],
            "top_emotion": classification["top_emotion"],
        },
        "emotions": classification["emotions"],
        "songs": songs,
    }
