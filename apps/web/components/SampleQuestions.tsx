"use client";

import type { Language } from "@/lib/types";

export const SAMPLE_QUESTIONS: { text: string; lang: Language; label?: string }[] = [
  { text: "what type of mountain is Mount Fuji", lang: "en" },
  { text: "who was Bridget Moynahan married to", lang: "en" },
  { text: "what is Kinsey most known for", lang: "en" },
  { text: "definition of philosophy", lang: "en" },
  { text: "என்ன வகையான மலை எம்டி ஃபுஜி?", lang: "ta", label: "தமிழ்" },
];

export function SampleQuestions({
  disabled,
  onPick,
}: {
  disabled?: boolean;
  onPick: (text: string, lang: Language) => void;
}) {
  return (
    <div className="mt-8">
      <p className="font-mono text-[11px] tracking-wide text-muted">Try one</p>
      <ol className="mt-3 space-y-2">
        {SAMPLE_QUESTIONS.map((sample, index) => (
          <li key={sample.text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(sample.text, sample.lang)}
              className="group flex w-full items-baseline gap-3 text-left disabled:cursor-not-allowed disabled:opacity-40"
            >
              <span className="font-mono text-[11px] tabular-nums text-muted">
                {String(index + 1).padStart(2, "0")}
              </span>
              {sample.label && (
                <span className="shrink-0 text-[11px] text-accent">{sample.label}</span>
              )}
              <span
                lang={sample.lang}
                className="text-[15px] leading-snug text-ink/80 underline-offset-4 group-hover:underline"
              >
                {sample.text}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
