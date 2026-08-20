"use client";

import { useEffect } from "react";

import { TracePanel } from "@/components/TracePanel";
import type { Citation, Trace } from "@/lib/types";

export function InfoSheet({
  open,
  onClose,
  citations,
  trace,
  budgetMs,
}: {
  open: boolean;
  onClose: () => void;
  citations: Citation[];
  trace: Trace | null;
  budgetMs: number;
}) {
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      <button
        type="button"
        className="absolute inset-0 bg-ink/35"
        aria-label="Close sources"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="info-sheet-title"
        className="absolute inset-x-0 bottom-0 max-h-[78vh] overflow-y-auto border-t border-rule bg-paper px-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-3"
      >
        <div className="mx-auto mb-3 h-1 w-10 rounded-full bg-rule" />
        <div className="mb-5 flex items-baseline justify-between gap-3">
          <h2 id="info-sheet-title" className="font-serif text-xl text-ink">
            Sources and timing
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="font-mono text-[11px] tracking-wide text-muted"
          >
            Close
          </button>
        </div>

        <section className="mb-6">
          <p className="mb-3 font-mono text-[11px] tracking-wide text-muted">Notes</p>
          {citations.length === 0 ? (
            <p className="text-sm leading-relaxed text-muted">No passages cited for this turn.</p>
          ) : (
            <ol className="space-y-3">
              {citations.map((citation, index) => (
                <li key={citation.chunk_id} className="flex gap-3 text-sm leading-relaxed text-ink/75">
                  <span className="font-mono tabular-nums text-muted">{index + 1}.</span>
                  <span>{citation.translated_text || citation.text}</span>
                </li>
              ))}
            </ol>
          )}
        </section>

        {trace ? (
          <TracePanel trace={trace} budgetMs={budgetMs} alwaysOpen />
        ) : (
          <p className="font-mono text-[11px] leading-relaxed text-muted">
            Timings show up after an answer lands.
          </p>
        )}
      </div>
    </div>
  );
}
