import { useMemo } from 'react';

// Two layers falling continuously from the top of the screen:
//   - flowers (pink ✿ blossoms)
//   - stars   (gold/pink ✦ sparkles)
// Each particle has a randomized column, fall speed, sway amount,
// rotation speed and size so the field never repeats visually.

const FLOWERS = 22;
const STARS = 18;

function randomParticle(i, kind) {
  return {
    key: `${kind}-${i}`,
    kind,
    left: `${Math.random() * 100}%`,
    size: kind === 'flower' ? 14 + Math.random() * 18 : 8 + Math.random() * 14,
    duration: `${12 + Math.random() * 14}s`,
    delay: `${-Math.random() * 20}s`,           // negative → start mid-fall
    sway: `${(Math.random() * 80 - 40).toFixed(0)}px`,
    spin: `${(Math.random() * 720 - 360).toFixed(0)}deg`,
  };
}

export default function FallingPetals() {
  const particles = useMemo(
    () => [
      ...Array.from({ length: FLOWERS }, (_, i) => randomParticle(i, 'flower')),
      ...Array.from({ length: STARS }, (_, i) => randomParticle(i, 'star')),
    ],
    []
  );

  return (
    <div className="petals" aria-hidden="true">
      {particles.map((p) => (
        <span
          key={p.key}
          className={`petals__item petals__item--${p.kind}`}
          style={{
            left: p.left,
            fontSize: `${p.size}px`,
            animationDuration: p.duration,
            animationDelay: p.delay,
            '--sway': p.sway,
            '--spin': p.spin,
          }}
        >
          {p.kind === 'flower' ? '✿' : '✦'}
        </span>
      ))}
    </div>
  );
}
