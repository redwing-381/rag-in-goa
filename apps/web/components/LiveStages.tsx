"use client";

import type { LiveStage } from "@/lib/types";

const PIPELINE: { name: string; label: string }[] = [
  { name: "stt", label: "speech" },
  { name: "embed_query", label: "embed" },
  { name: "dense_search", label: "dense" },
  { name: "sparse_search", label: "sparse" },
  { name: "llm", label: "answer" },
];

export function LiveStages({
  stages,
  pending,
  audio,
  retrievalMs,
}: {
  stages: LiveStage[];
  pending: boolean;
  audio: boolean;
  retrievalMs?: number | null;
}) {
  const byName = new Map(stages.map((stage) => [stage.name, stage]));
  const visible = PIPELINE.filter((item) => item.name !== "stt" || audio || byName.has("stt"));
  const active = pending ? visible.find((item) => !byName.has(item.name)) : undefined;
  if (!pending && stages.length === 0 && !retrievalMs) return null;

  return (
    <div className="flex flex-col gap-y-1.5 font-mono text-[11px] tracking-wide text-muted">
      {visible.map((item) => {
        const done = byName.get(item.name);
        const current = active?.name === item.name;
        return (
          <span
            key={item.name}
            className={
              done ? "text-ink" : current ? "text-accent" : "text-rule"
            }
          >
            {item.label}
            {current ? "…" : done ? ` ${done.duration_ms.toFixed(0)}ms` : ""}
          </span>
        );
      })}
      {retrievalMs != null && retrievalMs > 0 && !pending && (
        <span className="text-ink">retrieved {retrievalMs.toFixed(0)} ms</span>
      )}
    </div>
  );
}
