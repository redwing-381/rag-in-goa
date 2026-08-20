"use client";

import { type OrbState } from "orb-ui";
import { InfoGlyph, MicIcon, SpeakerIcon, SpinnerIcon, StopIcon } from "@/components/icons";

export function Composer({
  typed,
  onTyped,
  onSubmit,
  placeholder,
  disabled,
  orb,
  level,
  seconds,
  canPlay,
  playing,
  onPlay,
  onMic,
  onEndChat,
  error,
  hintAction,
}: {
  typed: string;
  onTyped: (value: string) => void;
  onSubmit: () => void;
  placeholder: string;
  disabled: boolean;
  orb: OrbState;
  level: number;
  seconds: number;
  canPlay: boolean;
  playing: boolean;
  onPlay: () => void;
  onMic: () => void;
  onEndChat: () => void;
  error: string | null;
  hintAction?: { label: string; onClick: () => void };
}) {
  const listening = orb === "listening";
  const thinking = orb === "thinking";
  const speaking = orb === "speaking";
  const inSession = listening || thinking || speaking;

  return (
    <div className="mx-auto w-full max-w-xl">
      <div className="flex items-center gap-2.5">
        <label className="flex min-h-14 min-w-0 flex-1 items-center rounded-2xl border border-rule bg-card px-4 shadow-[0_8px_24px_-18px_rgba(28,25,21,0.4)]">
          <input
            value={typed}
            onChange={(event) => onTyped(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") onSubmit();
            }}
            placeholder={placeholder}
            disabled={disabled}
            className="min-w-0 flex-1 bg-transparent py-3.5 text-base text-ink placeholder:text-muted/70 focus:outline-none disabled:opacity-40"
          />
          <button
            type="button"
            onClick={onPlay}
            disabled={!canPlay}
            aria-label={playing ? "Stop listening" : "Listen to the answer"}
            className="ml-2 shrink-0 p-1.5 text-muted disabled:opacity-30"
          >
            <SpeakerIcon />
          </button>
        </label>
        <button
          type="button"
          onClick={onMic}
          disabled={thinking || speaking || (orb === "idle" && disabled)}
          aria-label={listening ? "Tap to stop" : "Tap to speak"}
          className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-full text-white shadow-[0_8px_20px_-8px_rgba(156,66,33,0.8)] transition disabled:opacity-40 ${
            listening ? "bg-ink" : "bg-accent"
          }`}
          style={
            listening
              ? { transform: `scale(${1 + Math.min(level, 1) * 0.1})` }
              : undefined
          }
        >
          {thinking ? <SpinnerIcon /> : listening ? <StopIcon /> : <MicIcon />}
        </button>
      </div>
      {error && <p className="mt-2 text-center text-sm text-accent">{error}</p>}
      <div className="mt-3 flex items-center justify-center gap-2 px-1">
        {hintAction ? (
          <button
            type="button"
            onClick={hintAction.onClick}
            aria-label={hintAction.label}
            className="shrink-0 text-muted"
          >
            <InfoGlyph />
          </button>
        ) : (
          <span className="shrink-0 text-muted">
            <InfoGlyph />
          </span>
        )}
        <p className="min-w-0 text-[12px] leading-relaxed text-muted">
          {listening
            ? `${seconds.toFixed(1)}s · tap the mic to stop, or pause two seconds to send.`
            : thinking
              ? "Looking that up. The answer will write itself above."
              : speaking
                ? "Playing the answer. Tap the speaker to stop."
                : "Tap to stop when you finish, or pause two seconds to send."}
          {inSession && (
            <>
              {" "}
              <button
                type="button"
                onClick={onEndChat}
                className="underline-offset-2 hover:text-ink hover:underline"
              >
                End voice chat
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
