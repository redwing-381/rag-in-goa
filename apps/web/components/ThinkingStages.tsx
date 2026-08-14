"use client";

import { useEffect, useState } from "react";

type Mode = "audio" | "text";

const AUDIO_STAGES = ["Hearing you", "Finding passages", "Writing the answer"] as const;
const TEXT_STAGES = ["Finding passages", "Writing the answer"] as const;

/** Advance every 900ms so a ~3s LLM wait does not look like a hang. */
const TICK_MS = 900;

export function ThinkingStages({ mode }: { mode: Mode }) {
  const stages = mode === "audio" ? AUDIO_STAGES : TEXT_STAGES;
  const [index, setIndex] = useState(0);

  useEffect(() => {
    setIndex(0);
    const timer = window.setInterval(() => {
      setIndex((current) => Math.min(current + 1, stages.length - 1));
    }, TICK_MS);
    return () => window.clearInterval(timer);
  }, [mode, stages.length]);

  return (
    <div className="rounded-2xl border border-edge bg-surface/60 p-5">
      <ol className="space-y-2.5">
        {stages.map((label, i) => {
          const done = i < index;
          const active = i === index;
          return (
            <li key={label} className="flex items-center gap-3 text-sm">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  active ? "bg-accent" : done ? "bg-mint" : "bg-white/20"
                }`}
              />
              <span className={active ? "text-white" : done ? "text-white/50" : "text-white/30"}>
                {label}
                {active ? "…" : ""}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
