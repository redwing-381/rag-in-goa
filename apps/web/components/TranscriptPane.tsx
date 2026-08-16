"use client";

import { useEffect, useRef } from "react";

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
}: {
  turns: SessionTurn[];
  liveYou: string;
  liveAgent: string;
  liveStatus: string | null;
  language: string;
  refused: boolean;
  live: boolean;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  const empty = turns.length === 0 && !liveYou && !liveAgent && !liveStatus;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length, liveYou, liveAgent, liveStatus]);

  return (
    <section className="flex min-h-0 flex-[3] flex-col overflow-y-auto px-6 py-6">
      {empty ? (
        <div className="m-auto max-w-md text-center">
          <p className="text-lg text-white/70">Talk like you would to a person.</p>
          <p className="mt-2 text-sm text-white/40">
            Pause when you are done. You will see your words first, then the
            answer as it is written — the voice follows a beat later.
          </p>
        </div>
      ) : (
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-5">
          {turns.map((turn) => (
            <TurnBlock
              key={turn.id}
              you={turn.transcript || turn.query}
              agent={turn.answer}
              language={turn.lang}
              refused={turn.refused}
            />
          ))}
          {live && (
            <TurnBlock
              you={liveYou}
              agent={liveAgent}
              language={language}
              refused={refused}
              live
              status={liveStatus}
            />
          )}
          <div ref={bottom} />
        </div>
      )}
    </section>
  );
}

function TurnBlock({
  you,
  agent,
  language,
  refused,
  live,
  status,
}: {
  you: string;
  agent: string;
  language: string;
  refused: boolean;
  live?: boolean;
  status?: string | null;
}) {
  return (
    <div className="space-y-3">
      {you && (
        <div className="ml-8 rounded-2xl rounded-tr-md bg-white/6 px-4 py-3">
          <p className="text-[11px] font-semibold tracking-wider text-white/35 uppercase">
            You
          </p>
          <p className="mt-0.5 text-base leading-relaxed text-white/80">{you}</p>
        </div>
      )}
      {(agent || status || live) && (
        <div
          className={`mr-8 rounded-2xl rounded-tl-md px-4 py-3 ${
            refused ? "bg-warm/10" : "bg-accent/10"
          }`}
        >
          <p
            className={`text-[11px] font-semibold tracking-wider uppercase ${
              refused ? "text-warm" : "text-white/35"
            }`}
          >
            {refused ? "Declined" : "Agent"}
          </p>
          {status && !agent && (
            <p className="mt-1 text-base text-white/50 italic">{status}</p>
          )}
          {agent && (
            <p className="mt-0.5 text-lg leading-relaxed text-white/95" lang={language}>
              {agent}
              {live && <span className="ml-0.5 inline-block animate-pulse">▍</span>}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export function CitationStrip({ response }: { response: AskResponse }) {
  if (response.citations.length === 0) return null;
  return (
    <div className="mx-auto mt-4 w-full max-w-2xl">
      <p className="mb-2 text-[11px] font-semibold tracking-wider text-white/30 uppercase">
        Sources
      </p>
      <ul className="space-y-1.5">
        {response.citations.map((citation) => (
          <li
            key={citation.chunk_id}
            className="line-clamp-2 rounded-lg border border-edge/60 bg-raised/40 px-3 py-2 text-xs text-white/55"
          >
            {citation.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
