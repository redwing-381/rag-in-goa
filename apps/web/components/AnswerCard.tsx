"use client";

import { useEffect, useState } from "react";

import { speak, stopSpeaking, voiceFor } from "@/lib/speak";
import { type AskResponse, LANGUAGES, REFUSAL_COPY } from "@/lib/types";

/**
 * The answer, or the refusal.
 *
 * A refusal is presented as a legitimate outcome rather than an error, because
 * knowing when not to answer is the point: roughly 45% of the corpus is labelled
 * unanswerable, so a system that always produces prose would be wrong constantly
 * while looking confident.
 */
export function AnswerCard({
  response,
  muted,
  onMutedChange,
}: {
  response: AskResponse;
  muted: boolean;
  onMutedChange: (muted: boolean) => void;
}) {
  const language = LANGUAGES.find((entry) => entry.code === response.answer_language);
  const [voicesReady, setVoicesReady] = useState(false);
  const canSpeak = !response.refused && voicesReady && Boolean(voiceFor(response.answer_language));

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const mark = () => setVoicesReady(window.speechSynthesis.getVoices().length > 0);
    mark();
    window.speechSynthesis.addEventListener("voiceschanged", mark);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", mark);
  }, []);

  useEffect(() => {
    if (muted || response.refused || !voicesReady) {
      stopSpeaking();
      return;
    }
    speak(response.answer, response.answer_language);
    return () => stopSpeaking();
  }, [response, muted, voicesReady]);

  return (
    <section
      className={`rounded-2xl border p-6 ${
        response.refused ? "border-warm/30 bg-warm/5" : "border-edge bg-surface/60"
      }`}
    >
      {response.transcript && (
        <div className="mb-5 border-l-2 border-accent/40 pl-3">
          <p className="text-[11px] font-semibold tracking-wider text-white/35 uppercase">
            You said
          </p>
          <p className="mt-0.5 text-white/70 italic">&ldquo;{response.transcript}&rdquo;</p>
        </div>
      )}

      {response.refused ? (
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-warm" />
            <h2 className="text-sm font-semibold tracking-wide text-warm uppercase">
              Declined to answer
            </h2>
          </div>
          <p
            className="text-lg leading-relaxed text-white/90"
            lang={response.answer_language}
          >
            {response.answer}
          </p>
          {response.refusal_reason && (
            <p className="mt-3 text-sm text-white/50">
              {REFUSAL_COPY[response.refusal_reason]}
            </p>
          )}
        </div>
      ) : (
        <div>
          <div className="mb-3 flex items-center justify-between gap-4">
            <h2 className="text-sm font-semibold tracking-wide text-white/50 uppercase">
              Answer{language && language.code !== "en" ? ` · ${language.native}` : ""}
            </h2>
            <div className="flex items-center gap-2">
              <CopyButton text={response.answer} />
              {canSpeak && (
                <MuteButton muted={muted} onChange={onMutedChange} />
              )}
              <ConfidenceChip value={response.confidence} />
            </div>
          </div>
          <p
            className="text-lg leading-relaxed text-white/95"
            lang={response.answer_language}
          >
            {response.answer}
          </p>
        </div>
      )}

      {response.groundedness && <Groundedness report={response.groundedness} />}

      {response.citations.length > 0 && <Citations response={response} />}
    </section>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          setCopied(false);
        }
      }}
      className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-white/50
        transition hover:border-white/25 hover:text-white/80"
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

function MuteButton({ muted, onChange }: { muted: boolean; onChange: (muted: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => {
        if (!muted) stopSpeaking();
        onChange(!muted);
      }}
      className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-white/50
        transition hover:border-white/25 hover:text-white/80"
    >
      {muted ? "muted" : "speaking"}
    </button>
  );
}

function ConfidenceChip({ value }: { value: number }) {
  const tone =
    value >= 0.7 ? "text-mint bg-mint/10" : value >= 0.4 ? "text-warm bg-warm/10" : "text-white/50 bg-white/5";
  return (
    <span className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium ${tone}`}>
      confidence {(value * 100).toFixed(0)}%
    </span>
  );
}

function Groundedness({ report }: { report: NonNullable<AskResponse["groundedness"]> }) {
  return (
    <div className="mt-5 rounded-xl border border-edge/60 bg-raised/40 p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] font-semibold tracking-wider text-white/40 uppercase">
          Groundedness check
        </p>
        <span className={`text-xs font-medium ${report.grounded ? "text-mint" : "text-warm"}`}>
          {report.grounded ? "supported" : "not supported"} · {report.score.toFixed(2)}
        </span>
      </div>
      {report.invalid_citations.length > 0 && (
        <p className="mt-2 text-xs text-warm/80">
          Invented citations, dropped: {report.invalid_citations.join(", ")}
        </p>
      )}
      {report.unsupported_sentences.length > 0 && (
        <ul className="mt-2 space-y-1">
          {report.unsupported_sentences.map((sentence) => (
            <li key={sentence} className="text-xs text-warm/80">
              unsupported: &ldquo;{sentence}&rdquo;
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function firstSentences(text: string, count = 2): string {
  const parts = text.split(/(?<=[.!?।॥])\s+/).filter(Boolean);
  if (parts.length <= count) return text.trim();
  return parts.slice(0, count).join(" ").trim();
}

function Citations({ response }: { response: AskResponse }) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="mt-5">
      <p className="mb-2 text-[11px] font-semibold tracking-wider text-white/35 uppercase">
        Sources ({response.citations.length})
      </p>
      <ul className="space-y-2">
        {response.citations.map((citation) => {
          const expanded = open === citation.chunk_id;
          const body = citation.translated_text ?? citation.text;
          const preview = firstSentences(body, 2);
          return (
            <li key={citation.chunk_id} className="rounded-xl border border-edge/60 bg-raised/40">
              <div className="px-3 py-2.5">
                <p className="text-sm leading-relaxed text-white/70">{preview}</p>
                {preview !== body.trim() && (
                  <button
                    type="button"
                    onClick={() => setOpen(expanded ? null : citation.chunk_id)}
                    className="mt-2 text-[11px] text-accent-soft hover:text-white"
                  >
                    {expanded ? "Hide passage" : "Show passage"}
                  </button>
                )}
              </div>
              {expanded && (
                <div className="border-t border-edge/60 px-3 py-3">
                  <p className="text-sm leading-relaxed text-white/75">{body}</p>
                  <p className="mt-2 text-[11px] text-white/25">
                    doc {citation.doc_id}
                  </p>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
