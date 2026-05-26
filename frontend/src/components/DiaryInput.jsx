import { useState } from 'react';

const PLACEHOLDER =
  "today was a lot... write everything you'd say if no one was reading.";

// Format today's date like a real journal header: "Tuesday, 26 May 2026"
function formatToday() {
  return new Date().toLocaleDateString('en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

export default function DiaryInput({ onSubmit, loading }) {
  const [text, setText] = useState('');

  function submit(e) {
    e.preventDefault();
    if (text.trim().length < 10 || loading) return;
    onSubmit(text.trim());
  }

  return (
    <form className="diary" onSubmit={submit}>
      <div className="diary__date">{formatToday()}</div>
      <label className="diary__label" htmlFor="diary-entry">
        Dear Diary,
      </label>
      <textarea
        id="diary-entry"
        className="diary__textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        rows={8}
        disabled={loading}
      />
      <div className="diary__footer">
        <span className="diary__count">{text.length} / 4000</span>
        <button
          type="submit"
          className="diary__button"
          disabled={loading || text.trim().length < 10}
        >
          {loading ? 'putting pen to paper' : 'Generate verse'}
        </button>
      </div>
    </form>
  );
}
