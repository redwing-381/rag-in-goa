"use client";

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

const COPY: Record<OrbState, string> = {
  idle: "tap to talk",
  listening: "listening…",
  thinking: "on it…",
  speaking: "speaking…",
};

export function AgentOrb({
  state,
  level,
  seconds,
  disabled,
  onToggle,
  onStop,
  showStop,
}: {
  state: OrbState;
  level: number;
  seconds: number;
  disabled?: boolean;
  onToggle: () => void;
  onStop?: () => void;
  showStop?: boolean;
}) {
  const listening = state === "listening";
  const scale = listening ? 1 + Math.min(level, 1) * 0.18 : 1;
  const tone =
    state === "listening"
      ? "bg-warm text-ink pulse-ring"
      : state === "speaking"
        ? "bg-mint text-ink orb-speak"
        : state === "thinking"
          ? "bg-accent text-white"
          : "bg-accent text-white hover:bg-accent-soft";

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        type="button"
        disabled={disabled || state === "thinking" || state === "speaking"}
        onClick={onToggle}
        aria-label={listening ? "Stop listening" : "Start listening"}
        className={`relative flex h-24 w-24 items-center justify-center rounded-full transition
          disabled:cursor-not-allowed disabled:opacity-50 ${tone}`}
        style={{ transform: `scale(${scale})` }}
      >
        {state === "thinking" ? <Spinner /> : listening ? <Wave /> : <MicIcon />}
      </button>
      <div className="text-center">
        <p className={`text-sm ${listening ? "tabular-nums text-warm" : "text-white/50"}`}>
          {listening ? `${seconds.toFixed(1)}s` : COPY[state]}
        </p>
        {listening && (
          <p className="text-[11px] text-white/35">2s pause to send · say “goodbye” to end</p>
        )}
      </div>
      {showStop && onStop && (
        <button
          type="button"
          onClick={onStop}
          aria-label="End chat"
          className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-warm/50
            bg-warm/10 px-3.5 py-1.5 text-xs font-medium text-warm
            transition hover:border-warm hover:bg-warm/20"
        >
          <StopIcon />
          End chat
        </button>
      )}
    </div>
  );
}

function StopIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3" strokeLinecap="round" />
    </svg>
  );
}

function Wave() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="8" width="3" height="8" rx="1" />
      <rect x="10.5" y="4" width="3" height="16" rx="1" />
      <rect x="17" y="8" width="3" height="8" rx="1" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" fill="none" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" />
    </svg>
  );
}
