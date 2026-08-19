"use client";

import { useState } from "react";

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
  const [open, setOpen] = useState(false);
  const retrieval = trace.spans.filter((span) => !EXTERNAL_STAGES.has(span.name));
  const external = trace.spans.filter((span) => EXTERNAL_STAGES.has(span.name));
  const slowest = Math.max(...trace.spans.map((span) => span.duration_ms), 1);

  const withinBudget = trace.retrieval_ms <= budgetMs;

  return (
    <section>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-baseline justify-between gap-3 text-left"
      >
        <h2 className="font-mono text-[11px] tracking-wide text-muted">
          How long that took
          <span className={`ml-2 ${withinBudget ? "text-mint" : "text-accent"}`}>
            · {trace.retrieval_ms.toFixed(0)} ms retrieval
          </span>
        </h2>
        <span className="text-[11px] text-muted">{open ? "hide" : "show"}</span>
      </button>

      {!open && (
        <p className="mt-1 text-xs text-muted">
          {withinBudget ? "Inside the 200 ms budget." : "Over the 200 ms budget."}
          {trace.llm_ttft_ms !== null && ` Generation ${trace.llm_ttft_ms.toFixed(0)} ms.`}
        </p>
      )}

      {open && (
        <>
      <header className="mb-4 mt-4 flex items-baseline justify-between gap-3">
        <p className="font-mono text-[11px] tracking-wide text-muted">Latency trace</p>
        <code className="font-mono text-[11px] text-muted">{trace.request_id}</code>
      </header>

      <div className="mb-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div>
          <span className="text-muted">retrieval </span>
          <span className={withinBudget ? "font-medium text-mint" : "font-medium text-accent"}>
            {trace.retrieval_ms.toFixed(1)} ms
          </span>
          <span className="text-muted"> / {budgetMs.toFixed(0)} ms budget</span>
        </div>
        {trace.llm_ttft_ms !== null && (
          <div>
            <span className="text-muted">generation </span>
            <span className="font-medium text-ink">{trace.llm_ttft_ms.toFixed(0)} ms</span>
          </div>
        )}
        <div>
          <span className="text-muted">end to end </span>
          <span className="font-medium text-ink">{trace.total_ms.toFixed(0)} ms</span>
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
        <div className="mt-5 border border-accent/30 bg-accent/5 p-3">
          <p className="mb-1.5 font-mono text-[11px] tracking-wide text-accent">Ran degraded</p>
          <ul className="space-y-1 text-sm text-ink/80">
            {trace.degradations.map((item) => (
              <li key={item}>{DEGRADATION_COPY[item] ?? item}</li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-muted">
            The deadline dropped optional stages rather than overrun. Accuracy fell in a
            declared order; the budget held.
          </p>
        </div>
      )}
        </>
      )}
    </section>
  );
}

function BudgetBar({ used, budget }: { used: number; budget: number }) {
  const ratio = Math.min(used / budget, 1);
  const over = used > budget;

  return (
    <div className="mb-5">
      <div className="h-px w-full overflow-hidden bg-rule">
        <div
          className={`h-full ${over ? "bg-accent" : "bg-ink"}`}
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
      <p className="mb-2 font-mono text-[11px] tracking-wide text-muted">{title}</p>
      <ul className="space-y-1.5">
        {spans.map((span, index) => (
          <li key={`${span.name}-${index}`} className="grid grid-cols-[1fr_auto] items-center gap-3">
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="truncate text-sm text-ink">
                  {STAGE_LABELS[span.name] ?? span.name}
                </span>
                <StageMeta metadata={span.metadata} />
              </div>
              <div className="mt-1 h-px w-full overflow-hidden bg-rule">
                <div
                  className="h-full bg-ink/50"
                  style={{ width: `${Math.max((span.duration_ms / slowest) * 100, 1)}%` }}
                />
              </div>
            </div>
            <span className="font-mono text-sm tabular-nums text-muted">
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
    <span className="shrink-0 font-mono text-[11px] text-muted">
      {entries.map(([key, value]) => `${key}=${value}`).join(" ")}
    </span>
  );
}
