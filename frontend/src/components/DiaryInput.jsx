import { useState } from 'react';

const PLACEHOLDER =
  "today was a lot... write everything you'd say if no one was reading.";

export default function DiaryInput({ onSubmit, loading }) {
  const [text, setText] = useState('');

  function submit(e) {
    e.preventDefault();
    if (text.trim().length < 10 || loading) return;
    onSubmit(text.trim());
  }

  return (
    <form className="diary" onSubmit={submit}>
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
          {loading ? 'writing...' : 'Generate verse'}
        </button>
      </div>
    </form>
  );
}
