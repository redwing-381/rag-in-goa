"use client";

import type { Language } from "@/lib/types";

export const SAMPLE_QUESTIONS: { text: string; lang: Language; label: string }[] = [
  { text: "what type of mountain is Mount Fuji", lang: "en", label: "English" },
  { text: "who was Bridget Moynahan married to", lang: "en", label: "English" },
  { text: "माउंट फ़ूजी किस प्रकार का पहाड़ है?", lang: "hi", label: "हिन्दी" },
  { text: "দর্শনের সংজ্ঞা কী?", lang: "bn", label: "বাংলা" },
  { text: "என்ன வகையான மலை எம்டி ஃபுஜி?", lang: "ta", label: "தமிழ்" },
  { text: "किन्से कशासाठी सर्वाधिक प्रसिद्ध आहेत?", lang: "mr", label: "मराठी" },
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
          <li key={`${sample.lang}-${sample.text}`}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(sample.text, sample.lang)}
              className="group flex min-h-11 w-full items-start gap-3 text-left disabled:cursor-not-allowed disabled:opacity-40"
            >
              <span className="mt-0.5 font-mono text-[11px] tabular-nums text-muted">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="min-w-0">
                <span className="block text-[11px] text-accent">{sample.label}</span>
                <span
                  lang={sample.lang}
                  className="mt-0.5 block text-[15px] leading-snug break-words text-ink/80 underline-offset-4 group-hover:underline"
                >
                  {sample.text}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
