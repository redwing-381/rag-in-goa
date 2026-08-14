"use client";

import type { Trace } from "@/lib/types";

/**
 * Per-stage timings for the request just served.
 *
 * This is the same Trace object that backs the published latency numbers, so what
 * the demo shows and what the report claims come from one source. Retrieval and
 * generation are separated deliberately: the 200ms budget covers retrieval, which
 * we control, while the LLM is a network call to someone else's GPU and is reported
 * rather than promised.
 */

const STAGE_LABELS: Record<string, string> = {
  stt: "speech to text",
  stt_error: "speech to text failed",
  input_gate: "input guardrail",
  embed_query: "embed query",
  dense_search: "dense search (HNSW)",
  sparse_search: "lexical search (BM25)",
  fusion: "rank fusion (RRF)",
  rerank: "cross-encoder rerank",
  expand_context: "expand to parent context",
  llm: "answer generation",
  output_gate: "groundedness check",
  refused: "refusal",
};

const DEGRADATION_COPY: Record<string, string> = {
  rerank_skipped: "Reranking skipped to stay inside the budget",
  sparse_skipped: "Lexical leg skipped to stay inside the budget",
  dense_ef_reduced: "Dense search ran at reduced accuracy to stay inside the budget",
};

/** Stages that are ours to keep under 200ms, versus the upstream model call. */
const EXTERNAL_STAGES = new Set(["stt", "stt_error", "llm"]);

export function TracePanel({ trace, budgetMs }: { trace: Trace; budgetMs: number }) {
  const retrieval = trace.spans.filter((span) => !EXTERNAL_STAGES.has(span.name));
  const external = trace.spans.filter((span) => EXTERNAL_STAGES.has(span.name));
  const slowest = Math.max(...trace.spans.map((span) => span.duration_ms), 1);

  const withinBudget = trace.retrieval_ms <= budgetMs;

  return (
    <section className="rounded-2xl border border-edge bg-surface/60 p-5">
      <header className="mb-4 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-wide text-white/80 uppercase">
          Latency trace
        </h2>
        <code className="text-[11px] text-white/35">{trace.request_id}</code>
      </header>

      <div className="mb-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div>
          <span className="text-white/45">retrieval </span>
          <span className={withinBudget ? "font-semibold text-mint" : "font-semibold text-warm"}>
            {trace.retrieval_ms.toFixed(1)} ms
          </span>
          <span className="text-white/35"> / {budgetMs.toFixed(0)} ms budget</span>
        </div>
        {trace.llm_ttft_ms !== null && (
          <div>
            <span className="text-white/45">generation </span>
            <span className="font-semibold text-white/85">
              {trace.llm_ttft_ms.toFixed(0)} ms
            </span>
          </div>
        )}
        <div>
          <span className="text-white/45">end to end </span>
          <span className="font-semibold text-white/85">{trace.total_ms.toFixed(0)} ms</span>
        </div>
      </div>

      <BudgetBar used={trace.retrieval_ms} budget={budgetMs} />

      <StageList title="Retrieval — inside the budget" spans={retrieval} slowest={slowest} />
      {external.length > 0 && (
        <StageList
          title="External calls — measured, not budgeted"
          spans={external}
          slowest={slowest}
        />
      )}

      {trace.degradations.length > 0 && (
        <div className="mt-5 rounded-xl border border-warm/25 bg-warm/5 p-3">
          <p className="mb-1.5 text-xs font-semibold tracking-wide text-warm uppercase">
            Ran degraded
          </p>
          <ul className="space-y-1 text-sm text-white/70">
            {trace.degradations.map((item) => (
              <li key={item}>{DEGRADATION_COPY[item] ?? item}</li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-white/40">
            The deadline dropped optional stages rather than overrun. Accuracy fell in a
            declared order; the budget held.
          </p>
        </div>
      )}
    </section>
  );
}

function BudgetBar({ used, budget }: { used: number; budget: number }) {
  const ratio = Math.min(used / budget, 1);
  const over = used > budget;

  return (
    <div className="mb-5">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/8">
        <div
          className={`h-full rounded-full ${over ? "bg-warm" : "bg-mint"}`}
          style={{ width: `${Math.max(ratio * 100, 1.5)}%` }}
        />
      </div>
    </div>
  );
}

function StageList({
  title,
  spans,
  slowest,
}: {
  title: string;
  spans: Trace["spans"];
  slowest: number;
}) {
  if (spans.length === 0) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 text-[11px] font-semibold tracking-wider text-white/35 uppercase">
        {title}
      </p>
      <ul className="space-y-1.5">
        {spans.map((span, index) => (
          <li key={`${span.name}-${index}`} className="grid grid-cols-[1fr_auto] items-center gap-3">
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="truncate text-sm text-white/80">
                  {STAGE_LABELS[span.name] ?? span.name}
                </span>
                <StageMeta metadata={span.metadata} />
              </div>
              <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full bg-accent/60"
                  style={{ width: `${Math.max((span.duration_ms / slowest) * 100, 1)}%` }}
                />
              </div>
            </div>
            <span className="text-sm tabular-nums text-white/55">
              {span.duration_ms < 1
                ? `${span.duration_ms.toFixed(2)} ms`
                : `${span.duration_ms.toFixed(1)} ms`}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StageMeta({ metadata }: { metadata: Record<string, string | number | boolean> }) {
  const entries = Object.entries(metadata).filter(([, value]) => value !== "" && value !== null);
  if (entries.length === 0) return null;

  return (
    <span className="shrink-0 text-[11px] text-white/30">
      {entries.map(([key, value]) => `${key}=${value}`).join(" ")}
    </span>
  );
}
