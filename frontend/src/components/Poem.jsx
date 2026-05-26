// The unfolded letter — paper image background, scrollable, the
// verse fades in stanza-by-stanza after the unfold animation.

export default function Poem({ text, theme, cluster }) {
  const stanzas = text.split(/\n{2,}/).map((s) => s.split('\n'));

  return (
    <div className="letter-stage">
      <article className="letter__paper">
        <div className="letter__scroll">
          <div className="poem__meta">
            <span className="poem__theme">{theme?.name}</span>
            <span className="poem__dot">·</span>
            <span className="poem__cluster">{cluster?.label}</span>
            <span className="poem__dot">·</span>
            <span className="poem__emotion">{cluster?.top_emotion}</span>
          </div>

          <div className="poem__body">
            {stanzas.map((lines, i) => (
              <p key={i} className="poem__stanza">
                {lines.map((line, j) => (
                  <span key={j} className="poem__line">
                    {line}
                  </span>
                ))}
              </p>
            ))}
          </div>
        </div>
      </article>
    </div>
  );
}
