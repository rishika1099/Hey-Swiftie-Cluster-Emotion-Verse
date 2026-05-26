// Shown after a poem is generated but before it's revealed.
// An envelope rises out of the diary page, gently bouncing, with a "click
// to open" hint underneath. Clicking it triggers the letter reveal.

export default function Envelope({ onOpen }) {
  function handleKey(e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onOpen();
    }
  }

  return (
    <div className="envelope-stage">
      <button
        type="button"
        className="envelope-card"
        onClick={onOpen}
        onKeyDown={handleKey}
        aria-label="Open the letter"
      >
        <div className="envelope-card__body">
          <div className="envelope-card__flap" />
          <div className="envelope-card__seal">💌</div>
        </div>
      </button>
      <p className="envelope-hint">click to open ✨</p>
    </div>
  );
}
