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
    <div className="w-full max-w-md px-6">
      <p className="mb-2 text-[11px] font-semibold tracking-wider text-white/35 uppercase">
        Try one of these
      </p>
      <ul className="flex flex-wrap gap-2">
        {SAMPLE_QUESTIONS.map((sample) => (
          <li key={sample.text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(sample.text, sample.lang)}
              className="rounded-full border border-edge bg-raised/50 px-3 py-1.5 text-left text-xs
                text-white/70 transition hover:border-accent/40 hover:text-white
                disabled:cursor-not-allowed disabled:opacity-40"
            >
              {sample.label && (
                <span className="mr-1.5 text-accent-soft">{sample.label}</span>
              )}
              <span lang={sample.lang}>{sample.text}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
