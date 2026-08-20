"use client";

import { Orb, type OrbState } from "orb-ui";

export type { OrbState };

function copy(
  state: OrbState,
  seconds: number,
  recover: boolean,
): { title: string; detail: string } {
  if (state === "listening") {
    return {
      title: `${seconds.toFixed(1)}s · tap to stop`,
      detail: "Or pause two seconds to send.",
    };
  }
  if (state === "thinking") {
    return {
      title: "Looking that up",
      detail: "The answer will write itself above. End the chat to cancel.",
    };
  }
  if (state === "speaking") {
    return {
      title: "Playing the answer",
      detail: "Stop cuts the voice. Tap the bars when you want to ask again.",
    };
  }
  if (state === "error") {
    return {
      title: "Tap to try again",
      detail: "The microphone did not start.",
    };
  }
  if (recover) {
    return {
      title: "Tap to start",
      detail: "Tap to stop when you finish, or pause two seconds to send.",
    };
  }
  return {
    title: "Tap to start",
    detail: "Tap to stop when you finish, or pause two seconds to send.",
  };
}

export function AgentOrb({
  state,
  level,
  seconds,
  disabled,
  recover,
  onStart,
  onStop,
  onStopSpeaking,
  onEndChat,
}: {
  state: OrbState;
  level: number;
  seconds: number;
  disabled?: boolean;
  recover?: boolean;
  onStart: () => void;
  onStop: () => void;
  onStopSpeaking: () => void;
  onEndChat: () => void;
}) {
  const listening = state === "listening";
  const speaking = state === "speaking";
  const thinking = state === "thinking";
  const inSession = listening || speaking || thinking;
  const inputVolume = listening ? Math.min(Math.max(level, 0), 1) : 0;
  const outputVolume = speaking ? Math.min(Math.max(Math.max(level, 0.35), 0), 1) : 0;
  const words = copy(state, seconds, Boolean(recover));

  return (
    <div className="flex w-full max-w-md flex-col items-center">
      <div className="bars-well" data-state={state}>
        <Orb
          theme="bars"
          state={state}
          signal={{
            state,
            inputVolume,
            outputVolume,
            volume: listening ? inputVolume : outputVolume,
          }}
          size={220}
          onStart={onStart}
          onStop={onStop}
          disabled={disabled || thinking || speaking}
          aria-label={listening ? "Tap to stop" : "Tap to start"}
        />
      </div>
      <p
        className={`mt-3 text-center font-serif text-base leading-snug sm:text-lg sm:leading-none ${
          listening ? "font-mono text-sm tabular-nums text-accent sm:text-base" : "text-ink"
        }`}
      >
        {words.title}
      </p>
      <p className="mt-1.5 text-center text-sm leading-relaxed text-muted">{words.detail}</p>
      {inSession && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
          {speaking && (
            <button
              type="button"
              onClick={onStopSpeaking}
              className="font-mono text-[12px] tracking-wide text-accent underline-offset-4 hover:underline"
            >
              Stop
            </button>
          )}
          <button
            type="button"
            onClick={onEndChat}
            className="font-mono text-[12px] tracking-wide text-muted underline-offset-4 hover:text-ink hover:underline"
          >
            End voice chat
          </button>
        </div>
      )}
    </div>
  );
}
