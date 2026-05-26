import { useState } from 'react';
import DiaryInput from './components/DiaryInput.jsx';
import Poem from './components/Poem.jsx';
import Envelope from './components/Envelope.jsx';
import FloatingSongs from './components/FloatingSongs.jsx';
import FallingPetals from './components/FallingPetals.jsx';

export default function App() {
  // stage: 'idle' | 'loading' | 'envelope' | 'open'
  const [stage, setStage] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(diary) {
    setStage('loading');
    setError(null);
    setResult(null);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diary }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      setStage('envelope');
    } catch (e) {
      setError(e.message);
      setStage('idle');
    }
  }

  function openLetter() {
    setStage('open');
  }

  return (
    <div className="app">
      <FallingPetals />
      {stage === 'open' && result?.songs && <FloatingSongs songs={result.songs} />}

      <header className="app__header">
        <h1 className="app__title">Dear Diary,</h1>
        <h2 className="app__subtitle">Love, Taylor 💌</h2>
        <p className="app__tagline">
          tell me about your day — i'll write you back in verse
        </p>
      </header>

      <main className="app__main">
        <DiaryInput onSubmit={handleSubmit} loading={stage === 'loading'} />

        {error && <div className="app__error">{error}</div>}

        {stage === 'envelope' && result && (
          <Envelope onOpen={openLetter} />
        )}

        {stage === 'open' && result && (
          <section className="app__result">
            <Poem
              key={result.poem}
              text={result.poem}
              theme={result.theme}
              cluster={result.cluster}
            />
          </section>
        )}
      </main>
    </div>
  );
}
