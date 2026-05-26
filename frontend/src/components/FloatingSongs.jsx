import { useEffect, useRef } from 'react';

// Each recommended song is a small pink card that drifts slowly around
// the viewport, bouncing off the edges. You can:
//   • drag it anywhere with the mouse
//   • click it (without dragging) to open a Spotify search in a new tab
//
// We don't have Spotify track IDs in the dataset, so we can't `<iframe>`
// embed a real player — search is the next best thing. If the dataset
// ever gains track IDs, swap the URL for /embed/track/<id>.

// Starting "home" positions as fractions of viewport — left & right gutters
// only so they don't crowd the centered poem on first paint.
const HOMES = [
  [0.05, 0.10], [0.85, 0.12],
  [0.02, 0.35], [0.86, 0.38],
  [0.05, 0.62], [0.84, 0.65],
  [0.08, 0.85], [0.78, 0.88],
];

const DRIFT_SPEED = 14;     // px/s baseline — slow, ambient drift
const FLING_CAP = 250;      // max post-drag velocity (also limits drift)
const CLICK_THRESHOLD = 5;  // px before a press becomes a drag


function initItem(i) {
  const [hx, hy] = HOMES[i % HOMES.length];
  const W = typeof window !== 'undefined' ? window.innerWidth : 1200;
  const H = typeof window !== 'undefined' ? window.innerHeight : 800;
  const angle = Math.random() * Math.PI * 2;
  return {
    el: null,
    x: hx * W,
    y: hy * H,
    vx: Math.cos(angle) * DRIFT_SPEED,
    vy: Math.sin(angle) * DRIFT_SPEED,
    dragging: false,
    moved: false,
    offsetX: 0,
    offsetY: 0,
    lastT: 0,
    lastPx: 0,
    lastPy: 0,
  };
}


export default function FloatingSongs({ songs }) {
  // Physics state lives in a ref (one entry per song) so animation
  // updates don't trigger React re-renders.
  const itemsRef = useRef(null);
  if (itemsRef.current === null) {
    itemsRef.current = songs.map((_, i) => initItem(i));
  }

  // Whenever the song list changes (new diary entry), reset positions
  // but preserve any DOM bindings still attached.
  useEffect(() => {
    const old = itemsRef.current || [];
    itemsRef.current = songs.map((_, i) => ({
      ...initItem(i),
      el: old[i]?.el ?? null,
    }));
  }, [songs]);

  // The drift loop. One rAF for all bubbles.
  useEffect(() => {
    let last = performance.now();
    let raf = 0;
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const W = window.innerWidth;
      const H = window.innerHeight;
      for (const item of itemsRef.current || []) {
        if (!item.el || item.dragging) continue;
        item.x += item.vx * dt;
        item.y += item.vy * dt;
        const w = item.el.offsetWidth;
        const h = item.el.offsetHeight;
        if (item.x < 0)        { item.x = 0;       item.vx = Math.abs(item.vx); }
        if (item.y < 0)        { item.y = 0;       item.vy = Math.abs(item.vy); }
        if (item.x + w > W)    { item.x = W - w;   item.vx = -Math.abs(item.vx); }
        if (item.y + h > H)    { item.y = H - h;   item.vy = -Math.abs(item.vy); }
        item.el.style.transform = `translate(${item.x}px, ${item.y}px)`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ─── Pointer handlers ───────────────────────────────────────────
  function handleDown(i, e) {
    const item = itemsRef.current[i];
    item.dragging = true;
    item.moved = false;
    item.offsetX = e.clientX - item.x;
    item.offsetY = e.clientY - item.y;
    item.lastT = e.timeStamp;
    item.lastPx = e.clientX;
    item.lastPy = e.clientY;
    item.vx = 0;
    item.vy = 0;
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function handleMove(i, e) {
    const item = itemsRef.current[i];
    if (!item.dragging) return;
    const nx = e.clientX - item.offsetX;
    const ny = e.clientY - item.offsetY;
    if (!item.moved && Math.hypot(nx - item.x, ny - item.y) > CLICK_THRESHOLD) {
      item.moved = true;
    }
    const dt = Math.max(0.001, (e.timeStamp - item.lastT) / 1000);
    item.vx = (e.clientX - item.lastPx) / dt;
    item.vy = (e.clientY - item.lastPy) / dt;
    item.lastT = e.timeStamp;
    item.lastPx = e.clientX;
    item.lastPy = e.clientY;
    item.x = nx;
    item.y = ny;
    item.el.style.transform = `translate(${item.x}px, ${item.y}px)`;
  }

  function handleUp(i, e, song) {
    const item = itemsRef.current[i];
    if (!item.dragging) return;
    item.dragging = false;
    item.vx = Math.max(-FLING_CAP, Math.min(FLING_CAP, item.vx || 0));
    item.vy = Math.max(-FLING_CAP, Math.min(FLING_CAP, item.vy || 0));
    // If the press barely moved, treat it as a click → open Spotify.
    // Prefer a direct track URL when the backend matched the song against
    // the Spotify catalogue; otherwise fall back to a search query.
    if (!item.moved) {
      const url = song.spotify_id
        ? `https://open.spotify.com/track/${song.spotify_id}`
        : `https://open.spotify.com/search/${encodeURIComponent(
            `${song.track_title} Taylor Swift`
          )}`;
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  }

  // ─── Render ─────────────────────────────────────────────────────
  return (
    <div className="floats">
      {songs.map((song, i) => (
        <div
          key={`${song.album_name}-${song.track_title}-${i}`}
          ref={(el) => {
            if (!itemsRef.current[i]) itemsRef.current[i] = initItem(i);
            itemsRef.current[i].el = el;
            if (el) {
              el.style.transform =
                `translate(${itemsRef.current[i].x}px, ${itemsRef.current[i].y}px)`;
            }
          }}
          className="floats__item"
          onPointerDown={(e) => handleDown(i, e)}
          onPointerMove={(e) => handleMove(i, e)}
          onPointerUp={(e) => handleUp(i, e, song)}
          onPointerCancel={(e) => handleUp(i, e, song)}
          role="button"
          tabIndex={0}
          aria-label={`${song.track_title} from ${song.album_name} — drag to move, click to open in Spotify`}
        >
          <div
            className="floats__card"
            title="drag me · click to open in Spotify"
          >
            {song.cover ? (
              <img
                className="floats__cover"
                src={song.cover}
                alt=""
                draggable={false}
              />
            ) : (
              <div className="floats__cover floats__cover--placeholder">♪</div>
            )}
            <div className="floats__text">
              <div className="floats__title">{song.track_title}</div>
              <div className="floats__album">{song.album_name}</div>
            </div>
            <div className="floats__match">
              {Math.round(song.similarity * 100)}%
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
