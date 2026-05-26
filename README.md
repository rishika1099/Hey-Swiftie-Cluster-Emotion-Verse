# 💌 Dear Diary, Love Taylor

**🌸 Live at [dear-diary-love-taylor.vercel.app](https://dear-diary-love-taylor.vercel.app/)**

Tell the diary about your day. A fine-tuned BERT model traces your
emotions across seven dimensions, K-Means matches them against 867
clustered Taylor Swift songs, and GPT-4o-mini writes you back an original
verse in her lyrical style. Your letter arrives in a pink envelope you
click to open — wrinkled coffee-stained paper, hand-lettered fonts, and
eight draggable Spotify-linked song bubbles floating around the page,
ranked by how closely they match your mood.

- **Frontend**: Vite + React on Vercel — custom hand-lettered fonts,
  falling petals, drifting song bubbles you can drag around and click
  to open in Spotify.
- **Backend**: FastAPI on Hugging Face Spaces. DistilRoBERTa extracts
  your emotional profile; K-Means clusters all 867 Taylor Swift songs by
  emotion; GPT-4o-mini writes the verse in her style; a Spotify enrichment
  pass joins each recommended song to its real Spotify track ID.

## Project layout

```
.
├── backend/
│   └── main.py                      # FastAPI app — /api/generate
├── src/
│   ├── data_processing.py           # pipeline step 1 — clean lyrics
│   ├── sentiment_analyzer.py        # pipeline step 2 — BERT emotions
│   ├── clustering.py                # pipeline step 3 — K-Means
│   ├── diary_classifier.py          # match diary → cluster
│   ├── theme_manager.py             # album-aesthetic themes
│   └── letter_generator_openai.py   # OpenAI verse generation
├── convert_kaggle_dataset.py        # turn Kaggle zip into CSV + covers
├── deduplicate_songs.py             # 1 row per song
├── enrich_spotify.py                # join with Spotify catalogue
├── frontend/
│   ├── public/
│   │   ├── fonts/                   # all self-hosted fonts
│   │   └── paper.png                # letter background image
│   └── src/
│       ├── App.jsx
│       ├── styles.css
│       └── components/
│           ├── DiaryInput.jsx
│           ├── Envelope.jsx         # click-to-open envelope
│           ├── Poem.jsx
│           ├── FloatingSongs.jsx    # draggable song bubbles
│           └── FallingPetals.jsx
├── Dockerfile                       # for Hugging Face Spaces
├── requirements.txt
└── README.md
```

## Local setup

### 1. Get the datasets

You need two Kaggle downloads:

- [Taylor Swift All Lyrics](https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums) — lyrics + album covers
- [Taylor Swift Spotify data](https://www.kaggle.com/datasets/jarredpriester/taylor-swift-spotify-dataset) — track IDs + popularity

Save them as `archive.zip` and `data/raw/taylor_swift_spotify.csv`
respectively in the project root.

### 2. Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt torch
```

(`torch` is listed separately so you can swap in a GPU build if you want.)

### 3. Run the pipeline

```bash
python convert_kaggle_dataset.py            # CSV + cover_art/
python src/data_processing.py
python src/sentiment_analyzer.py            # ~5–10 min on CPU
python src/clustering.py
python deduplicate_songs.py
python enrich_spotify.py                    # adds spotify_id to each song
```

### 4. Frontend deps

```bash
cd frontend
npm install
cd ..
```

### 5. Set your OpenAI key

```bash
export OPENAI_API_KEY=sk-...
```

### 6. Run it

Two terminals:

```bash
# terminal 1
uvicorn backend.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```

Open http://localhost:5173.

## Deployment (free forever)

The stack splits cleanly: static frontend on Vercel, ML backend on
Hugging Face Spaces.

### Backend → Hugging Face Spaces (free CPU)

1. Create a new Space at https://huggingface.co/new-space with **SDK = Docker**.
2. Push this repo to the Space's git remote.
3. Set `OPENAI_API_KEY` as a Space secret.
4. Wait ~10 min for the first build. The Space auto-restarts on each push.

The Dockerfile at the repo root handles everything. The pipeline outputs
(`data/processed/`, `models/`, `cover_art/`) get baked into the image, so
make sure they exist locally before you push.

### Frontend → Vercel (free)

1. Push this repo to GitHub.
2. Import on https://vercel.com with **Root Directory = frontend**.
3. Vercel auto-detects Vite. No build config needed.
4. Edit `frontend/vercel.json` — replace `YOUR-HF-SPACE` with your actual
   Space subdomain so `/api/*` proxies to your backend.

## API

`POST /api/generate`

```json
{ "diary": "today was a lot..." }
```

returns:

```json
{
  "poem": "stanza...\n\nstanza...",
  "theme": { "name": "folklore", "primary": "#...", ... },
  "cluster": { "id": 3, "label": "Heartbreak Ballads", "top_emotion": "sadness" },
  "emotions": { "joy": 0.12, "sadness": 0.71, ... },
  "songs": [
    {
      "track_title": "the last great american dynasty",
      "album_name": "folklore",
      "similarity": 0.92,
      "cover": "data:image/jpeg;base64,...",
      "spotify_id": "3qHGAY8tnXQjxX9YjV4cdr"
    }
  ]
}
```

`GET /api/health` reports whether models loaded.

## License

Code: MIT. Lyrics, album covers, song metadata, Taylor's likeness all
belong to their respective owners. This is a fan project.
