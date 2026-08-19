"use client";

import { useEffect, useRef, type ReactNode } from "react";

import type { SessionTurn } from "@/lib/session";
import type { AskResponse } from "@/lib/types";

export function TranscriptPane({
  turns,
  liveYou,
  liveAgent,
  liveStatus,
  language,
  refused,
  live,
  emptyExtra,
}: {
  turns: SessionTurn[];
  liveYou: string;
  liveAgent: string;
  liveStatus: string | null;
  language: string;
  refused: boolean;
  live: boolean;
  emptyExtra?: ReactNode;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  const empty = turns.length === 0 && !liveYou && !liveAgent && !liveStatus;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length, liveYou, liveAgent, liveStatus]);

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-y-auto px-6 py-8 lg:px-10">
      {empty ? (
        <div className="m-auto w-full max-w-xl">
          <p className="font-serif text-2xl leading-snug text-ink">
            Ask a question the corpus can answer.
          </p>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-muted">
            Pause when you are done. Your words appear first, then the answer —
            the voice follows a beat later.
          </p>
          {emptyExtra}
        </div>
      ) : (
        <article className="mx-auto w-full max-w-xl">
          {turns.map((turn, index) => (
            <TurnBlock
              key={turn.id}
              index={index + 1}
              you={turn.transcript || turn.query}
              agent={turn.answer}
              language={turn.lang}
              refused={turn.refused}
            />
          ))}
          {live && (
            <TurnBlock
              index={turns.length + 1}
              you={liveYou}
              agent={liveAgent}
              language={language}
              refused={refused}
              live
              status={liveStatus}
            />
          )}
          <div ref={bottom} />
        </article>
      )}
    </section>
  );
}

function TurnBlock({
  index,
  you,
  agent,
  language,
  refused,
  live,
  status,
}: {
  index: number;
  you: string;
  agent: string;
  language: string;
  refused: boolean;
  live?: boolean;
  status?: string | null;
}) {
  return (
    <section className="border-b border-rule py-7 last:border-b-0">
      <p className="font-mono text-[11px] tracking-wide text-muted">
        {String(index).padStart(2, "0")}
      </p>
      {you && (
        <div className="mt-3">
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted">
            You
          </p>
          <p className="mt-1 text-[15px] leading-relaxed text-ink/80">{you}</p>
        </div>
      )}
      {(agent || status || live) && (
        <div className="mt-5">
          <p
            className={`text-[11px] font-medium uppercase tracking-[0.14em] ${
              refused ? "text-accent" : "text-muted"
            }`}
          >
            {refused ? "Declined" : "Answer"}
          </p>
          {status && !agent && (
            <p className="mt-2 font-serif text-lg italic leading-relaxed text-muted">
              {status}
            </p>
          )}
          {agent && (
            <p
              className={`mt-2 font-serif text-[1.2rem] leading-[1.65] ${
                refused ? "text-accent" : "text-ink"
              }`}
              lang={language}
            >
              {agent}
              {live && (
                <span className="ml-0.5 inline-block animate-pulse text-muted">▍</span>
              )}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export function CitationStrip({ response }: { response: AskResponse }) {
  if (response.citations.length === 0) return null;
  return (
    <footer className="mx-auto w-full max-w-xl border-t border-rule pt-4">
      <p className="mb-3 font-mono text-[11px] tracking-wide text-muted">Notes</p>
      <ol className="space-y-2">
        {response.citations.map((citation, index) => (
          <li key={citation.chunk_id} className="flex gap-3 text-xs leading-relaxed text-muted">
            <span className="font-mono tabular-nums text-ink/50">{index + 1}.</span>
            <span className="line-clamp-3">{citation.text}</span>
          </li>
        ))}
      </ol>
    </footer>
  );
}
