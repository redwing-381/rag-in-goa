"use client";

import { useEffect, useRef, type ReactNode } from "react";

import { InfoGlyph, SpeakerIcon } from "@/components/icons";
import type { SessionTurn } from "@/lib/session";
import type { Language } from "@/lib/types";

export function TranscriptPane({
  turns,
  liveYou,
  liveAgent,
  liveStatus,
  language,
  refused,
  live,
  emptyExtra,
  playingId,
  onListen,
  onInfo,
}: {
  turns: SessionTurn[];
  liveYou: string;
  liveAgent: string;
  liveStatus: string | null;
  language: string;
  refused: boolean;
  live: boolean;
  emptyExtra?: ReactNode;
  playingId?: string | null;
  onListen?: (id: string, text: string, lang: Language) => void;
  onInfo?: (id: string) => void;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  const empty = turns.length === 0 && !liveYou && !liveAgent && !liveStatus;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length, liveYou, liveAgent, liveStatus]);

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-5 sm:px-6 sm:py-8 lg:px-8">
      {empty ? (
        <div className="mx-auto w-full max-w-xl pb-6">{emptyExtra}</div>
      ) : (
        <article className="mx-auto w-full max-w-xl">
          {turns.map((turn, index) => (
            <TurnBlock
              key={turn.id}
              id={turn.id}
              index={index + 1}
              you={turn.transcript || turn.query}
              agent={turn.answer}
              language={turn.lang}
              refused={turn.refused}
              playing={playingId === turn.id}
              onListen={onListen}
              onInfo={onInfo}
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
  id,
  index,
  you,
  agent,
  language,
  refused,
  live,
  status,
  playing,
  onListen,
  onInfo,
}: {
  id?: string;
  index: number;
  you: string;
  agent: string;
  language: string;
  refused: boolean;
  live?: boolean;
  status?: string | null;
  playing?: boolean;
  onListen?: (id: string, text: string, lang: Language) => void;
  onInfo?: (id: string) => void;
}) {
  return (
    <section className="border-b border-rule py-5 last:border-b-0 sm:py-7">
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
            <>
              <p
                className={`mt-2 font-serif text-lg leading-[1.65] sm:text-[1.2rem] ${
                  refused ? "text-accent" : "text-ink"
                }`}
                lang={language}
              >
                {agent}
                {live && (
                  <span className="ml-0.5 inline-block animate-pulse text-muted">▍</span>
                )}
              </p>
              {!live && id && (
                <div className="mt-3 flex items-center gap-1">
                  {!refused && onListen && (
                    <button
                      type="button"
                      onClick={() => onListen(id, agent, language as Language)}
                      aria-label={playing ? "Stop listening" : "Listen to the answer"}
                      className={`inline-flex h-8 w-8 items-center justify-center rounded-full ${playing ? "text-accent" : "text-muted"}`}
                    >
                      <SpeakerIcon />
                    </button>
                  )}
                  {onInfo && (
                    <button
                      type="button"
                      onClick={() => onInfo(id)}
                      aria-label="Sources"
                      className="inline-flex h-8 items-center gap-1.5 rounded-full px-1.5 text-muted"
                    >
                      <InfoGlyph />
                      <span className="hidden font-mono text-[11px] tracking-wide lg:inline">
                        Source
                      </span>
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
