"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import { MicButton, useRecordingClock } from "@/components/MicButton";
import { SampleQuestions } from "@/components/SampleQuestions";
import { ThinkingStages } from "@/components/ThinkingStages";
import { TracePanel } from "@/components/TracePanel";
import { ApiError, askAudio, askText, health } from "@/lib/api";
import { MicRecorder, RecorderError, createLevelMeter } from "@/lib/recorder";
import { stopSpeaking } from "@/lib/speak";
import { type AskResponse, type HealthResponse, LANGUAGES, type Language } from "@/lib/types";

type Phase = "idle" | "recording" | "busy";

export default function Page() {
  const [service, setService] = useState<HealthResponse | null>(null);
  const [serviceError, setServiceError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [phase, setPhase] = useState<Phase>("idle");
  const [busyMode, setBusyMode] = useState<"audio" | "text">("text");
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [typed, setTyped] = useState("");
  const [muted, setMuted] = useState(false);

  const recorder = useRef<MicRecorder | null>(null);
  const meter = useRef<ReturnType<typeof createLevelMeter> | null>(null);
  const spoken = LANGUAGES.find((entry) => entry.code === language);

  useEffect(() => {
    health()
      .then(setService)
      .catch((err: unknown) =>
        setServiceError(
          err instanceof ApiError
            ? `API responded ${err.status}: ${err.message}`
            : "Could not reach the API. Is it running?",
        ),
      );
  }, []);

  const stopMeter = useCallback(() => {
    meter.current?.stop();
    meter.current = null;
    setLevel(0);
  }, []);

  const submitAudio = useCallback(
    async (wav: Blob) => {
      setBusyMode("audio");
      setPhase("busy");
      try {
        setResponse(await askAudio(wav, language));
      } catch (err) {
        setError(
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : "The request failed. Check that the API is still running.",
        );
      } finally {
        setPhase("idle");
      }
    },
    [language],
  );

  const stopRecording = useCallback(async () => {
    const active = recorder.current;
    if (!active?.active) return;
    stopMeter();
    setBusyMode("audio");
    setPhase("busy");
    try {
      await submitAudio(await active.stop());
    } catch (err) {
      setError(err instanceof RecorderError ? err.message : "Recording failed.");
      setPhase("idle");
    }
  }, [stopMeter, submitAudio]);

  const seconds = useRecordingClock(phase === "recording", () => void stopRecording());

  const startRecording = useCallback(async () => {
    setError(null);
    setResponse(null);
    stopSpeaking();
    const mic = new MicRecorder();
    recorder.current = mic;
    try {
      await mic.start();
      setPhase("recording");
      if (mic.mediaStream) {
        meter.current = createLevelMeter(mic.mediaStream);
        const tick = () => {
          if (!meter.current) return;
          setLevel(meter.current.read());
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }
    } catch (err) {
      setError(err instanceof RecorderError ? err.message : "Could not start recording.");
      setPhase("idle");
    }
  }, []);

  const submitQuery = useCallback(async (query: string, lang: Language) => {
    const text = query.trim();
    if (!text) return;
    setError(null);
    setResponse(null);
    stopSpeaking();
    setLanguage(lang);
    setTyped(text);
    setBusyMode("text");
    setPhase("busy");
    try {
      setResponse(await askText(text, lang));
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `${err.message} (HTTP ${err.status})`
          : "The request failed. Check that the API is still running.",
      );
    } finally {
      setPhase("idle");
    }
  }, []);

  const submitTyped = useCallback(() => {
    void submitQuery(typed, language);
  }, [typed, language, submitQuery]);

  return (
    <main className="mx-auto max-w-3xl px-5 py-12 sm:py-16">
      <header className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Ask out loud
        </h1>
        <p className="mt-2.5 max-w-2xl leading-relaxed text-white/50">
          Speak a question in any of five languages. It is transcribed and translated in one
          call, answered only from passages retrieved out of MS MARCO-XI, and refused outright
          when the corpus cannot support an answer.
        </p>
      </header>

      <section className="mb-8">
        <label className="mb-2 block text-[11px] font-semibold tracking-wider text-white/35 uppercase">
          I will speak
        </label>
        <div className="flex flex-wrap gap-2">
          {LANGUAGES.map((entry) => (
            <button
              key={entry.code}
              type="button"
              onClick={() => setLanguage(entry.code)}
              className={`rounded-full border px-3.5 py-1.5 text-sm transition ${
                language === entry.code
                  ? "border-accent bg-accent/15 text-white"
                  : "border-edge text-white/55 hover:border-white/25 hover:text-white/80"
              }`}
            >
              <span lang={entry.code}>{entry.native}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="mb-8 flex flex-col items-center gap-6 rounded-2xl border border-edge bg-surface/40 py-10">
        <MicButton
          state={phase}
          seconds={seconds}
          level={level}
          onStart={() => void startRecording()}
          onStop={() => void stopRecording()}
          disabled={!service?.ready}
        />
        {phase === "recording" && spoken && (
          <p className="text-sm text-white/55">
            Answering in <span lang={spoken.code}>{spoken.native}</span>
          </p>
        )}

        <div className="w-full max-w-md px-6">
          <div className="mb-3 flex items-center gap-3">
            <span className="h-px flex-1 bg-edge" />
            <span className="text-[11px] tracking-wider text-white/25 uppercase">or type</span>
            <span className="h-px flex-1 bg-edge" />
          </div>
          <div className="flex gap-2">
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") submitTyped();
              }}
              placeholder="what is a corporation"
              disabled={phase !== "idle" || !service?.ready}
              className="min-w-0 flex-1 rounded-xl border border-edge bg-raised/60 px-3.5 py-2.5 text-sm
                text-white/85 placeholder:text-white/25 focus:border-accent/60 focus:outline-none
                disabled:opacity-40"
            />
            <button
              type="button"
              onClick={submitTyped}
              disabled={phase !== "idle" || !typed.trim() || !service?.ready}
              className="shrink-0 rounded-xl bg-white/10 px-4 py-2.5 text-sm font-medium text-white/85
                transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-30"
            >
              Ask
            </button>
          </div>
        </div>

        {phase === "idle" && !response && !error && (
          <SampleQuestions
            disabled={!service?.ready}
            onPick={(text, lang) => void submitQuery(text, lang)}
          />
        )}
      </section>

      {error && (
        <div className="mb-8 rounded-2xl border border-warm/30 bg-warm/5 p-4">
          <p className="text-sm text-warm">{error}</p>
        </div>
      )}

      {phase === "busy" && !response && <ThinkingStages mode={busyMode} />}

      {response && (
        <div className="space-y-6">
          <AnswerCard response={response} muted={muted} onMutedChange={setMuted} />
          <TracePanel
            trace={response.trace}
            budgetMs={service?.retrieval_budget_ms ?? 200}
          />
        </div>
      )}

      <footer className="mt-14 border-t border-edge/60 pt-6 text-xs leading-relaxed text-white/25">
        <ServiceStatus service={service} error={serviceError} />
        <p className="mt-3">
          Built on MS MARCO-XI, which is released for non-commercial research use. Retrieval
          runs under a hard 200ms deadline and sheds optional stages rather than overrunning
          it; when that happens, the trace says so.
        </p>
      </footer>
    </main>
  );
}

function ServiceStatus({
  service,
  error,
}: {
  service: HealthResponse | null;
  error: string | null;
}) {
  if (error) {
    return (
      <p className="inline-flex items-center gap-2 text-warm">
        <span className="h-1.5 w-1.5 rounded-full bg-warm" />
        {error}
      </p>
    );
  }
  if (!service) {
    return <p>checking the service…</p>;
  }

  return (
    <p className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 text-white/35">
      <span className="inline-flex items-center gap-1.5">
        <span
          className={`h-1.5 w-1.5 rounded-full ${service.ready ? "bg-mint" : "bg-warm"}`}
        />
        {service.ready ? "ready" : "loading"}
      </span>
      {service.chunks !== null && (
        <span>{service.chunks.toLocaleString()} chunks</span>
      )}
      {service.docs !== null && <span>{service.docs.toLocaleString()} docs</span>}
      {service.strategy && <span>{service.strategy}</span>}
      {service.has_sparse && <span>hybrid</span>}
    </p>
  );
}
