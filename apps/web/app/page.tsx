"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import { MicButton, useRecordingClock } from "@/components/MicButton";
import { TracePanel } from "@/components/TracePanel";
import { ApiError, askAudio, askText, health } from "@/lib/api";
import { MicRecorder, RecorderError, createLevelMeter } from "@/lib/recorder";
import { type AskResponse, type HealthResponse, LANGUAGES, type Language } from "@/lib/types";

type Phase = "idle" | "recording" | "busy";

export default function Page() {
  const [service, setService] = useState<HealthResponse | null>(null);
  const [serviceError, setServiceError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [phase, setPhase] = useState<Phase>("idle");
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [typed, setTyped] = useState("");

  const recorder = useRef<MicRecorder | null>(null);
  const meter = useRef<ReturnType<typeof createLevelMeter> | null>(null);

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
    const mic = new MicRecorder();
    recorder.current = mic;
    try {
      await mic.start();
      setPhase("recording");
      // Read the stream the recorder already holds rather than opening the mic a
      // second time, so the button can react to the user's voice and make it
      // obvious that capture is live.
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

  const submitTyped = useCallback(async () => {
    const query = typed.trim();
    if (!query) return;
    setError(null);
    setResponse(null);
    setPhase("busy");
    try {
      setResponse(await askText(query, language));
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `${err.message} (HTTP ${err.status})`
          : "The request failed. Check that the API is still running.",
      );
    } finally {
      setPhase("idle");
    }
  }, [typed, language]);

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
        <ServiceStatus service={service} error={serviceError} />
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
        <p className="mt-2 text-xs text-white/30">
          The index is English. Speech is translated on the way in, and the answer comes back
          in the language you chose.
        </p>
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
                if (event.key === "Enter") void submitTyped();
              }}
              placeholder="what is a corporation"
              disabled={phase !== "idle" || !service?.ready}
              className="min-w-0 flex-1 rounded-xl border border-edge bg-raised/60 px-3.5 py-2.5 text-sm
                text-white/85 placeholder:text-white/25 focus:border-accent/60 focus:outline-none
                disabled:opacity-40"
            />
            <button
              type="button"
              onClick={() => void submitTyped()}
              disabled={phase !== "idle" || !typed.trim() || !service?.ready}
              className="shrink-0 rounded-xl bg-white/10 px-4 py-2.5 text-sm font-medium text-white/85
                transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-30"
            >
              Ask
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className="mb-8 rounded-2xl border border-warm/30 bg-warm/5 p-4">
          <p className="text-sm text-warm">{error}</p>
        </div>
      )}

      {phase === "busy" && !response && <Thinking />}

      {response && (
        <div className="space-y-6">
          <AnswerCard response={response} />
          <TracePanel
            trace={response.trace}
            budgetMs={service?.retrieval_budget_ms ?? 200}
          />
        </div>
      )}

      <footer className="mt-14 border-t border-edge/60 pt-6 text-xs leading-relaxed text-white/25">
        Built on MS MARCO-XI, which is released for non-commercial research use. Retrieval runs
        under a hard 200ms deadline and sheds optional stages rather than overrunning it; when
        that happens, the trace above says so.
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
      <p className="mt-4 inline-flex items-center gap-2 rounded-full border border-warm/30 bg-warm/5 px-3 py-1 text-xs text-warm">
        <span className="h-1.5 w-1.5 rounded-full bg-warm" />
        {error}
      </p>
    );
  }
  if (!service) {
    return <p className="mt-4 text-xs text-white/25">checking the service…</p>;
  }

  return (
    <p className="mt-4 inline-flex flex-wrap items-center gap-x-3 gap-y-1 rounded-full border border-edge bg-surface/60 px-3.5 py-1.5 text-xs text-white/45">
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

function Thinking() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((row) => (
        <div
          key={row}
          className="shimmer h-4 rounded-full bg-white/5"
          style={{ width: `${100 - row * 18}%` }}
        />
      ))}
    </div>
  );
}
