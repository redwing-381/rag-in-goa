export function HeroCard() {
  return (
    <div className="relative min-h-[13.5rem] overflow-hidden rounded-3xl bg-card px-5 py-6 shadow-[0_10px_30px_-18px_rgba(28,25,21,0.35)] sm:min-h-[15rem] sm:px-7 sm:py-7">
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <img
          src="/hero-goa.png"
          alt=""
          className="hero-goa h-full w-full object-cover object-[78%_50%]"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-card via-card/80 to-card/25" />
      </div>
      <div className="relative z-10 max-w-[17.5rem] sm:max-w-sm">
        <h1 className="font-serif text-[1.65rem] leading-[1.2] text-ink sm:text-3xl">
          Ask a question the corpus can answer.
        </h1>
        <p className="mt-3 text-[13.5px] leading-relaxed text-muted sm:text-sm">
          Type a question, or tap the mic to speak. Tap to stop, or pause two
          seconds to send. Use the speaker icon to listen to the answer.
        </p>
      </div>
    </div>
  );
}
