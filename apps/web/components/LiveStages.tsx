"use client";

import type { LiveStage } from "@/lib/types";

const PIPELINE: { name: string; label: string }[] = [
  { name: "stt", label: "speech to text" },
  { name: "embed_query", label: "embed" },
  { name: "dense_search", label: "dense" },
  { name: "sparse_search", label: "sparse" },
  { name: "rerank", label: "rerank" },
  { name: "llm", label: "llm" },
];

export function LiveStages({
  stages,
  pending,
  audio,
}: {
  stages: LiveStage[];
  pending: boolean;
  audio: boolean;
}) {
  const byName = new Map(stages.map((stage) => [stage.name, stage]));
  const visible = PIPELINE.filter((item) => item.name !== "stt" || audio || byName.has("stt"));
  const lastDone = [...visible].reverse().find((item) => byName.has(item.name));

  return (
    <section className="rounded-2xl border border-edge bg-surface/60 p-4">
      <p className="mb-3 text-[11px] font-semibold tracking-wider text-white/35 uppercase">
        Live retrieval
      </p>
      <ol className="space-y-2">
        {visible.map((item) => {
          const done = byName.get(item.name);
          const active = pending && !done && lastDone?.name !== item.name && isNext(visible, byName, item.name);
          return (
            <li key={item.name} className="flex items-center justify-between gap-3 text-sm">
              <span className="inline-flex items-center gap-2">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    done ? "bg-mint" : active ? "bg-accent" : "bg-white/20"
                  }`}
                />
                <span className={done ? "text-white/80" : active ? "text-white" : "text-white/30"}>
                  {item.label}
                  {active ? "…" : ""}
                </span>
              </span>
              <span className="tabular-nums text-xs text-white/40">
                {done ? `${done.duration_ms.toFixed(0)} ms` : ""}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function isNext(
  visible: { name: string }[],
  byName: Map<string, LiveStage>,
  name: string,
): boolean {
  const firstPending = visible.find((item) => !byName.has(item.name));
  return firstPending?.name === name;
}
