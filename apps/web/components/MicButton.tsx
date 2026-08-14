"use client";

import { useEffect, useRef, useState } from "react";

const MAX_SECONDS = 30; // Sarvam's sync endpoint limit

type State = "idle" | "recording" | "busy";

/**
 * Push to talk.
 *
 * Recording stops itself at 30 seconds because that is the provider's hard limit
 * for the synchronous endpoint - better to cut the clip and get an answer than to
 * send 40 seconds and be rejected after the user has finished speaking.
 */
export function MicButton({
  state,
  seconds,
  level,
  onStart,
  onStop,
  disabled,
}: {
  state: State;
  seconds: number;
  level: number;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}) {
  const recording = state === "recording";
  const remaining = MAX_SECONDS - seconds;

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        type="button"
        disabled={disabled || state === "busy"}
        onClick={recording ? onStop : onStart}
        aria-label={recording ? "Stop recording" : "Start recording"}
        className={`relative flex h-20 w-20 items-center justify-center rounded-full transition
          disabled:cursor-not-allowed disabled:opacity-40
          ${
            recording
              ? "pulse-ring bg-warm text-ink"
              : "bg-accent text-white hover:bg-accent-soft active:scale-95"
          }`}
        style={
          recording
            ? { transform: `scale(${1 + Math.min(level, 1) * 0.12})` }
            : undefined
        }
      >
        {state === "busy" ? <Spinner /> : recording ? <StopIcon /> : <MicIcon />}
      </button>

      <div className="text-center">
        {recording ? (
          <>
            <p className="text-sm font-medium tabular-nums text-warm">
              {seconds.toFixed(1)}s
            </p>
            <p className="text-xs text-white/35">
              {remaining <= 5
                ? `stopping in ${Math.max(remaining, 0).toFixed(0)}s`
                : "tap to stop"}
            </p>
          </>
        ) : (
          <p className="text-xs text-white/40">
            {state === "busy" ? "thinking…" : "tap and ask a question"}
          </p>
        )}
      </div>
    </div>
  );
}

/** Elapsed-time ticker plus the auto-stop at the provider's limit. */
export function useRecordingClock(active: boolean, onLimit: () => void) {
  const [seconds, setSeconds] = useState(0);
  const limitRef = useRef(onLimit);
  limitRef.current = onLimit;

  useEffect(() => {
    if (!active) {
      setSeconds(0);
      return;
    }
    const started = performance.now();
    const timer = window.setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      setSeconds(elapsed);
      if (elapsed >= MAX_SECONDS) limitRef.current();
    }, 100);
    return () => window.clearInterval(timer);
  }, [active]);

  return seconds;
}

function MicIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v3" strokeLinecap="round" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" fill="none" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" />
    </svg>
  );
}
