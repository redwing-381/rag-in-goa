"use client";

import { ChevronIcon } from "@/components/icons";
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
    <div className="mt-7">
      <p className="px-1 font-sans text-[15px] text-ink">Try one</p>
      <ol className="mt-1">
        {SAMPLE_QUESTIONS.map((sample, index) => (
          <li key={`${sample.lang}-${sample.text}`} className="border-b border-rule/80 last:border-b-0">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(sample.text, sample.lang)}
              className="flex min-h-14 w-full items-start gap-3 py-3.5 text-left disabled:cursor-not-allowed disabled:opacity-40"
            >
              <span className="mt-0.5 w-6 shrink-0 font-mono text-[12px] tabular-nums text-muted">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[12px] text-accent">{sample.label}</span>
                <span lang={sample.lang} className="mt-0.5 block text-[15px] leading-snug text-ink">
                  {sample.text}
                </span>
              </span>
              <ChevronIcon className="mt-2.5 shrink-0 text-muted/70" />
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
