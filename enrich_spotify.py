"""
Join the deduplicated clustered dataset against the Spotify catalogue so
each row gets the official Spotify track name, album name, and track ID.

Matching strategy
-----------------
The lyrics dataset has track titles like ``MyBoyOnlyBreaksHisFavoriteToys``
and album folders like ``Red_TaylorsVersion_``. The Spotify dataset has
properly-spaced names with parenthetical qualifiers like
``Fortnight (feat. Post Malone)`` and ``Red (Taylor's Version)``.

For each lyrics row we compute a *canonical key* (lowercase, alphanumeric-
only) for both the track title and the album name. The Spotify side gets
the same key, **after** stripping parenthetical/bracketed qualifiers like
``(feat. ...)`` or ``[Deluxe]``. If a single Spotify track matches by
title+album we use that; otherwise we fall back to title-only match.

Output
------
``data/processed/taylor_swift_clustered_enriched.csv`` — the dedup CSV
plus three new columns: ``spotify_name``, ``spotify_album``, ``spotify_id``.
Rows with no Spotify match keep empty strings in those columns.
"""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CLUSTERED = ROOT / "data" / "processed" / "taylor_swift_clustered_dedup.csv"
CLUSTERED_FALLBACK = ROOT / "data" / "processed" / "taylor_swift_clustered.csv"
SPOTIFY = ROOT / "data" / "raw" / "taylor_swift_spotify.csv"
OUT = ROOT / "data" / "processed" / "taylor_swift_clustered_enriched.csv"


# Edition/version marker substrings (already lowercased + alphanumeric-only)
# that should be stripped after squashing to a comparable canonical key. The
# dataset folder names have these glued together (TaylorsVersion, FromTheVault,
# thelongpondstudiosessions, …) while Spotify has them in parentheses — so we
# strip them on BOTH sides after reducing to [a-z0-9] only.
_MARKERS = sorted(
    [
        "taylorsversion", "taylorversion",
        "fromthevault",
        "acousticversion", "acoustic",
        "10minuteversion", "minuteversion",
        "theshortfilm", "shortfilm",
        "deluxeedition", "deluxeversion", "deluxe",
        "tangerineedition", "webstoredeluxe",
        "3amedition", "thetildawnedition", "thetilldawnedition",
        "thelatenightedition",
        "targetexclusive", "applemusicexclusive", "recordstoredayexclusive",
        "physicalversion", "platinumedition",
        "thelongpondstudiosessions",
        "fromthedisneyspecial", "thedisneyspecial",
        "japaneseedition",
        "digitaldeluxe", "digitallyautographedfanedition",
        "heartshapedvinyl",
        "ep", "prologue", "bonustrack", "remix", "remixes",
        "livefromparis", "livefromsosfest", "livefromkansascity",
        "feat", "ft",  # featuring
    ],
    key=len,
    reverse=True,  # strip longest first so substrings don't shadow each other
)


def canonical(name: str) -> str:
    """Reduce a track or album name to a comparable canonical key.

    1. lowercase
    2. drop parenthetical/bracketed qualifiers
    3. reduce to [a-z0-9] only
    4. strip known glued edition/version markers
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    while True:
        new = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*", " ", s)
        if new == s:
            break
        s = new
    key = re.sub(r"[^a-z0-9]+", "", s)
    for m in _MARKERS:
        key = key.replace(m, "")
    return key


def main() -> None:
    src = CLUSTERED if CLUSTERED.exists() else CLUSTERED_FALLBACK
    if not src.exists():
        raise SystemExit(
            f"❌ Need the clustered CSV at {src}. Run clustering first."
        )
    if not SPOTIFY.exists():
        raise SystemExit(
            f"❌ Need the Spotify catalogue at {SPOTIFY}. "
            f"Save taylor_swift_spotify.csv there."
        )

    print(f"📥 Loading {src.name}")
    df = pd.read_csv(src)

    print(f"📥 Loading {SPOTIFY.name}")
    spo = pd.read_csv(SPOTIFY)

    # Build Spotify lookup indices
    spo["title_key"] = spo["name"].map(canonical)
    spo["album_key"] = spo["album"].map(canonical)

    # Title + album → row (preferred)
    by_title_album = {}
    # Title-only → list of rows (fallback)
    by_title = {}
    for _, row in spo.iterrows():
        if row["title_key"]:
            by_title_album.setdefault((row["title_key"], row["album_key"]), row)
            by_title.setdefault(row["title_key"], []).append(row)

    print(f"\n🔗 Matching {len(df)} rows against Spotify catalogue…")

    names, albums, ids, uris = [], [], [], []
    matched_strict = matched_loose = unmatched = 0

    for _, row in df.iterrows():
        tkey = canonical(row["track_title"])
        akey = canonical(row["album_name"])

        hit = by_title_album.get((tkey, akey))
        if hit is None and tkey in by_title:
            # Fallback — pick the most-popular candidate if Spotify gives us
            # popularity, otherwise the first.
            candidates = by_title[tkey]
            if "popularity" in candidates[0]:
                hit = max(candidates, key=lambda r: r.get("popularity", 0))
            else:
                hit = candidates[0]
            matched_loose += 1
        elif hit is not None:
            matched_strict += 1

        if hit is not None:
            names.append(hit["name"])
            albums.append(hit["album"])
            ids.append(hit["id"])
            uris.append(hit["uri"])
        else:
            names.append("")
            albums.append("")
            ids.append("")
            uris.append("")
            unmatched += 1

    df["spotify_name"] = names
    df["spotify_album"] = albums
    df["spotify_id"] = ids
    df["spotify_uri"] = uris

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    total = len(df)
    print(f"\n✅ Enriched dataset → {OUT}")
    print(f"   strict matches (title + album): {matched_strict}")
    print(f"   loose matches (title only):     {matched_loose}")
    print(f"   unmatched:                      {unmatched}  ({unmatched/total:.0%})")
    print()
    print("Sample of enriched rows:")
    sample = df[df["spotify_id"] != ""].sample(min(8, (df["spotify_id"] != "").sum()))
    print(sample[["track_title", "spotify_name", "spotify_album"]].to_string(index=False))


if __name__ == "__main__":
    main()
