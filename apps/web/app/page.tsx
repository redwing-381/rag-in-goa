"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AgentOrb, type OrbState } from "@/components/AgentOrb";
import { LiveStages } from "@/components/LiveStages";
import { useRecordingClock } from "@/components/MicButton";
import { SampleQuestions } from "@/components/SampleQuestions";
import { SidebarHistory } from "@/components/SidebarHistory";
import { CitationStrip, TranscriptPane } from "@/components/TranscriptPane";
import { TracePanel } from "@/components/TracePanel";
import { ApiError, askAudioStream, askStream, health } from "@/lib/api";
import { MicRecorder, RecorderError } from "@/lib/recorder";
import { CLOSING, OPENING, isEndCommand } from "@/lib/greetings";
import { loadTurns, newTurnId, saveTurns, type SessionTurn, type TurnKind } from "@/lib/session";
import {
  canSpeak,
  preloadVoices,
  setSarvamTts,
  setSpeakListener,
  speakNext,
  stopSpeaking,
  takeSpeakable,
  whenSpeechEnds,
} from "@/lib/speak";
import { END_SILENCE_MS, createVad } from "@/lib/vad";
import {
  type AskResponse,
  type HealthResponse,
  type HistoryTurn,
  LANGUAGES,
  type Language,
  type LiveStage,
  type StreamEvent,
} from "@/lib/types";

const MAX_SECONDS = 30;

export default function Page() {
  const [service, setService] = useState<HealthResponse | null>(null);
  const [serviceError, setServiceError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [orb, setOrb] = useState<OrbState>("idle");
  const [level, setLevel] = useState(0);
  const [muted, setMuted] = useState(false);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [turns, setTurns] = useState<SessionTurn[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draftYou, setDraftYou] = useState("");
  const [draftAgent, setDraftAgent] = useState("");
  const [draftRefused, setDraftRefused] = useState(false);
  const [stages, setStages] = useState<LiveStage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [audioTurn, setAudioTurn] = useState(false);
  const [finalResponse, setFinalResponse] = useState<AskResponse | null>(null);
  const [voicesReady, setVoicesReady] = useState(false);
  const [waitHint, setWaitHint] = useState<string | null>(null);
  const [conversing, setConversing] = useState(false);

  const recorder = useRef<MicRecorder | null>(null);
  const vad = useRef<ReturnType<typeof createVad> | null>(null);
  const heardSpeech = useRef(false);
  const sending = useRef(false);
  const autoListen = useRef(false);
  const greeted = useRef(false);
  const ending = useRef(false);
  const cycle = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const speakBuf = useRef("");
  const mutedRef = useRef(muted);
  const languageRef = useRef(language);
  const orbRef = useRef(orb);
  mutedRef.current = muted;
  languageRef.current = language;
  orbRef.current = orb;

  useEffect(() => {
    preloadVoices();
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const mark = () => setVoicesReady(window.speechSynthesis.getVoices().length > 0);
    mark();
    window.speechSynthesis.addEventListener("voiceschanged", mark);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", mark);
  }, []);

  useEffect(() => {
    setTurns(loadTurns());
    health()
      .then((status) => {
        setSarvamTts(Boolean(status.has_tts));
        setService(status);
      })
      .catch((err: unknown) =>
        setServiceError(
          err instanceof ApiError
            ? `API responded ${err.status}: ${err.message}`
            : "Could not reach the API. Is it running?",
        ),
      );
  }, []);

  useEffect(() => {
    saveTurns(turns);
  }, [turns]);

  useEffect(() => {
    setSpeakListener(() => setOrb("speaking"));
    return () => setSpeakListener(null);
  }, []);

  const historyForRequest = useCallback((): HistoryTurn[] => {
    return turns
      .filter((turn) => (turn.kind ?? "qa") === "qa" && turn.query)
      .slice(-3)
      .flatMap((turn) => [
        { role: "user" as const, text: turn.query },
        { role: "assistant" as const, text: turn.answer },
      ]);
  }, [turns]);

  const stopCapture = useCallback(() => {
    vad.current?.stop();
    vad.current = null;
    setLevel(0);
  }, []);

  const haltCapture = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    autoListen.current = false;
    sending.current = false;
    heardSpeech.current = false;
    stopCapture();
    recorder.current?.cancel();
    recorder.current = null;
    setStreaming(false);
    setWaitHint(null);
    setAudioTurn(false);
  }, [stopCapture]);

  const pushUserLine = useCallback((text: string) => {
    const turn: SessionTurn = {
      id: newTurnId(),
      query: text,
      transcript: text,
      answer: "",
      lang: languageRef.current,
      trace: null,
      refused: false,
      citations: [],
      kind: "qa",
    };
    setTurns((current) => [...current, turn]);
    setSelectedId(turn.id);
    setDraftYou(text);
  }, []);

  const pushAgentLine = useCallback((text: string, kind: TurnKind) => {
    const turn: SessionTurn = {
      id: newTurnId(),
      query: "",
      transcript: null,
      answer: text,
      lang: languageRef.current,
      trace: null,
      refused: false,
      citations: [],
      kind,
    };
    setTurns((current) => [...current, turn]);
    setSelectedId(turn.id);
  }, []);

  const deliverLine = useCallback(
    async (text: string, kind: TurnKind) => {
      pushAgentLine(text, kind);
      if (mutedRef.current) return;
      setOrb("speaking");
      speakNext(text, languageRef.current);
      await whenSpeechEnds();
    },
    [pushAgentLine],
  );

  const endConversation = useCallback(async () => {
    if (ending.current) return;
    ending.current = true;
    cycle.current += 1;
    const id = cycle.current;
    haltCapture();
    stopSpeaking();

    if (greeted.current) {
      await deliverLine(CLOSING[languageRef.current], "farewell");
    }
    if (id !== cycle.current) {
      ending.current = false;
      return;
    }
    greeted.current = false;
    setConversing(false);
    setOrb("idle");
    ending.current = false;
  }, [deliverLine, haltCapture]);

  const finishSpeakThenListen = useCallback(async (id: number) => {
    await whenSpeechEnds();
    if (id !== cycle.current) return;
    setOrb("idle");
    if (autoListen.current && service?.ready) {
      window.setTimeout(() => {
        if (id === cycle.current && orbRef.current === "idle") {
          void startListeningRef.current();
        }
      }, 200);
    }
  }, [service?.ready]);

  const consumeStream = useCallback(
    async (events: AsyncGenerator<StreamEvent>, queryHint: string, fromAudio: boolean) => {
      const id = cycle.current;
      setStreaming(true);
      setAudioTurn(fromAudio);
      setOrb("thinking");
      if (fromAudio && !draftYou) setWaitHint("Heard you. Transcribing…");
      setStages([]);
      setDraftAgent("");
      setDraftRefused(false);
      setFinalResponse(null);
      setSelectedId(null);
      speakBuf.current = "";
      if (!fromAudio) setDraftYou(queryHint);

      const flushSpeak = () => {
        const rest = speakBuf.current.trim();
        speakBuf.current = "";
        if (rest && !mutedRef.current) speakNext(rest, languageRef.current);
      };

      let last: AskResponse | null = null;
      try {
        for await (const event of events) {
          if (id !== cycle.current) return;
          if (event.type === "transcript") {
            setDraftYou(event.text);
            setWaitHint(null);
            if (isEndCommand(event.text)) {
              pushUserLine(event.text);
              abort.current?.abort();
              await endConversation();
              return;
            }
          } else if (event.type === "stage") {
            setStages((current) => [...current, event]);
          } else if (event.type === "token") {
            setDraftAgent((current) => current + event.text);
            if (!mutedRef.current) {
              speakBuf.current += event.text;
              const { chunks, rest } = takeSpeakable(speakBuf.current);
              speakBuf.current = rest;
              for (const chunk of chunks) speakNext(chunk, languageRef.current);
            }
          } else if (event.type === "final") {
            last = event.response;
          }
        }
      } finally {
        setStreaming(false);
      }

      if (id !== cycle.current) return;
      if (!last) return;

      if (last.refused) {
        stopSpeaking();
        speakBuf.current = "";
        setDraftAgent(last.answer);
        setDraftRefused(true);
      } else {
        setDraftAgent(last.answer);
        flushSpeak();
      }
      setFinalResponse(last);
      const turn: SessionTurn = {
        id: newTurnId(),
        query: last.transcript || queryHint,
        transcript: last.transcript,
        answer: last.answer,
        lang: last.answer_language,
        trace: last.trace,
        refused: last.refused,
        citations: last.citations,
      };
      setTurns((current) => [...current, turn]);
      setSelectedId(turn.id);

      const willSpeak = !last.refused && !mutedRef.current && canSpeak(last.answer_language);
      if (willSpeak) setOrb("speaking");
      await finishSpeakThenListen(id);
    },
    [endConversation, finishSpeakThenListen, pushUserLine],
  );

  const startListeningRef = useRef<() => Promise<void>>(async () => {});

  const sendAudio = useCallback(
    async (wav: Blob) => {
      cycle.current += 1;
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      try {
        await consumeStream(
          askAudioStream(wav, languageRef.current, historyForRequest(), controller.signal),
          draftYou || "…",
          true,
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : "The request failed. Check that the API is still running.",
        );
        setOrb("idle");
        setStreaming(false);
      }
    },
    [consumeStream, draftYou, historyForRequest],
  );

  const stopAndSend = useCallback(async () => {
    if (sending.current) return;
    const active = recorder.current;
    if (!active?.active) return;
    sending.current = true;
    stopCapture();
    try {
      if (!heardSpeech.current) {
        active.cancel();
        recorder.current = null;
        setOrb("idle");
        return;
      }
      setOrb("thinking");
      setWaitHint("Heard you. Transcribing…");
      const wav = await active.stop();
      recorder.current = null;
      await sendAudio(wav);
    } catch (err) {
      setError(err instanceof RecorderError ? err.message : "Recording failed.");
      setOrb("idle");
    } finally {
      sending.current = false;
    }
  }, [sendAudio, stopCapture]);

  const startListening = useCallback(async () => {
    if (!service?.ready || ending.current) return;
    setError(null);
    const id = cycle.current;
    if (!greeted.current) {
      greeted.current = true;
      autoListen.current = true;
      setConversing(true);
      await deliverLine(OPENING[languageRef.current], "greeting");
      if (id !== cycle.current || ending.current) return;
    } else {
      stopSpeaking();
    }
    heardSpeech.current = false;
    sending.current = false;
    autoListen.current = true;
    setConversing(true);
    const mic = new MicRecorder();
    recorder.current = mic;
    try {
      await mic.start();
      setOrb("listening");
      setDraftYou("");
      setDraftAgent("");
      setDraftRefused(false);
      setWaitHint(null);
      setFinalResponse(null);
      setStages([]);
      setSelectedId(null);
      if (mic.mediaStream) {
        vad.current = createVad(mic.mediaStream, {
          silenceMs: END_SILENCE_MS,
          onLevel: setLevel,
          onSpeechStart: () => {
            heardSpeech.current = true;
          },
          onSpeechEnd: () => {
            void stopAndSend();
          },
        });
      }
    } catch (err) {
      setError(err instanceof RecorderError ? err.message : "Could not start recording.");
      setOrb("idle");
    }
  }, [deliverLine, service?.ready, stopAndSend]);

  startListeningRef.current = startListening;

  const seconds = useRecordingClock(orb === "listening", () => void stopAndSend());

  useEffect(() => {
    if (orb === "listening" && seconds >= MAX_SECONDS) void stopAndSend();
  }, [orb, seconds, stopAndSend]);

  const toggleOrb = useCallback(() => {
    if (orb === "listening") {
      void stopAndSend();
      return;
    }
    if (orb === "idle") void startListening();
  }, [orb, startListening, stopAndSend]);

  const submitText = useCallback(
    async (query: string, lang: Language) => {
      const text = query.trim();
      if (!text) return;
      cycle.current += 1;
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      setLanguage(lang);
      languageRef.current = lang;
      setTyped(text);
      setError(null);
      stopSpeaking();
      autoListen.current = true;
      setConversing(true);
      if (isEndCommand(text)) {
        greeted.current = true;
        pushUserLine(text);
        await endConversation();
        return;
      }
      if (!greeted.current) {
        greeted.current = true;
        await deliverLine(OPENING[lang], "greeting");
        if (ending.current) return;
      }
      const events = askStream(text, lang, historyForRequest(), controller.signal);
      try {
        await consumeStream(events, text, false);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : "The request failed. Check that the API is still running.",
        );
        setOrb("idle");
        setStreaming(false);
      }
    },
    [consumeStream, deliverLine, endConversation, historyForRequest, pushUserLine],
  );

  const newChat = useCallback(() => {
    ending.current = true;
    cycle.current += 1;
    haltCapture();
    stopSpeaking();
    greeted.current = false;
    ending.current = false;
    setConversing(false);
    setOrb("idle");
    setTurns([]);
    setSelectedId(null);
    setDraftYou("");
    setDraftAgent("");
    setDraftRefused(false);
    setStages([]);
    setFinalResponse(null);
    setError(null);
    setTyped("");
  }, [haltCapture]);

  const live = orb !== "idle" || streaming;
  const inFlight = streaming || Boolean(waitHint);
  const selected = turns.find((turn) => turn.id === selectedId) ?? null;
  const refused = live ? draftRefused : Boolean(selected?.refused ?? draftRefused);
  const shownTrace = live ? finalResponse?.trace : (selected?.trace ?? finalResponse?.trace);
  const shownResponse: AskResponse | null = live
    ? finalResponse
    : selected && finalResponse && selected.id === selectedId
      ? finalResponse
      : selected
        ? {
            answer: selected.answer,
            refused: selected.refused,
            refusal_reason: null,
            citations: selected.citations,
            confidence: 0,
            transcript: selected.transcript,
            answer_language: selected.lang,
            groundedness: null,
            trace: selected.trace ?? {
              request_id: selected.id,
              spans: [],
              degradations: [],
              tool_calls: [],
              retrieval_ms: 0,
              llm_ttft_ms: null,
              total_ms: 0,
              cache_hit: false,
            },
          }
        : finalResponse;

  return (
    <div className="flex h-dvh overflow-hidden">
      <div className="hidden lg:block">
        <SidebarHistory
          turns={turns.filter((turn) => (turn.kind ?? "qa") === "qa")}
          selectedId={selectedId}
          onSelect={(id) => {
            if (live) return;
            setSelectedId(id);
          }}
          onNewChat={newChat}
        />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-edge px-5 py-3 lg:hidden">
          <p className="text-sm text-white/70">Voice agent</p>
          <button
            type="button"
            onClick={newChat}
            className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-white/60"
          >
            New chat
          </button>
        </header>

        <TranscriptPane
          turns={turns}
          liveYou={inFlight ? draftYou : ""}
          liveAgent={inFlight ? draftAgent : ""}
          liveStatus={
            inFlight && !draftAgent
              ? waitHint || (draftYou ? "Looking that up…" : null)
              : inFlight && draftAgent && orb === "thinking"
                ? "Voice is catching up…"
                : null
          }
          language={selected?.lang ?? language}
          refused={refused}
          live={inFlight}
        />
        {shownResponse && !streaming && (
          <div className="shrink-0 px-6 pb-3">
            <CitationStrip response={shownResponse} />
          </div>
        )}

        <div className="flex min-h-[220px] flex-[2] flex-col items-center justify-center gap-5 border-t border-edge bg-surface/30 px-5 py-5">
          <AgentOrb
            state={orb}
            level={level}
            seconds={seconds}
            disabled={!service?.ready}
            onToggle={toggleOrb}
            onStop={endConversation}
            showStop={conversing || orb !== "idle" || streaming}
          />

          <div className="flex flex-wrap items-center justify-center gap-2">
            {LANGUAGES.map((entry) => (
              <button
                key={entry.code}
                type="button"
                onClick={() => setLanguage(entry.code)}
                className={`rounded-full border px-3 py-1 text-xs transition ${
                  language === entry.code
                    ? "border-accent bg-accent/15 text-white"
                    : "border-edge text-white/50 hover:border-white/25 hover:text-white/80"
                }`}
              >
                <span lang={entry.code}>{entry.native}</span>
              </button>
            ))}
            <button
              type="button"
              onClick={() => {
                if (!muted) stopSpeaking();
                setMuted((value) => !value);
              }}
              className="rounded-full border border-edge px-3 py-1 text-xs text-white/50
                hover:border-white/25 hover:text-white/80"
            >
              {muted ? "muted" : "voice on"}
            </button>
          </div>
          {voicesReady && !muted && !canSpeak(language) && (
            <p className="max-w-sm text-center text-[11px] text-white/40">
              This browser has no {LANGUAGES.find((entry) => entry.code === language)?.label ?? language}{" "}
              voice. The answer stays on screen.
            </p>
          )}

          <div className="flex w-full max-w-lg gap-2">
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void submitText(typed, language);
              }}
              placeholder="or type a follow-up"
              disabled={orb !== "idle" || !service?.ready}
              className="min-w-0 flex-1 rounded-xl border border-edge bg-raised/60 px-3.5 py-2.5 text-sm
                text-white/85 placeholder:text-white/25 focus:border-accent/60 focus:outline-none
                disabled:opacity-40"
            />
            <button
              type="button"
              onClick={() => void submitText(typed, language)}
              disabled={orb !== "idle" || !typed.trim() || !service?.ready}
              className="shrink-0 rounded-xl bg-white/10 px-4 py-2.5 text-sm font-medium text-white/85
                transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-30"
            >
              Ask
            </button>
          </div>

          {orb === "idle" && turns.length === 0 && !error && (
            <SampleQuestions
              disabled={!service?.ready}
              onPick={(text, lang) => void submitText(text, lang)}
            />
          )}

          {error && <p className="max-w-md text-center text-sm text-warm">{error}</p>}
        </div>
      </div>

      <aside className="hidden h-full w-[280px] shrink-0 flex-col overflow-y-auto border-l border-edge bg-surface/40 p-3 lg:flex">
        <LiveStages stages={stages} pending={streaming} audio={audioTurn} />
        {shownTrace && (
          <div className="mt-3">
            <TracePanel
              trace={shownTrace}
              budgetMs={service?.retrieval_budget_ms ?? 200}
            />
          </div>
        )}
        <div className="mt-auto pt-6 text-[11px] leading-relaxed text-white/30">
          <ServiceStatus service={service} error={serviceError} />
        </div>
      </aside>
    </div>
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
  if (!service) return <p>checking the service…</p>;
  return (
    <p className="inline-flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="inline-flex items-center gap-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${service.ready ? "bg-mint" : "bg-warm"}`} />
        {service.ready ? "ready" : "loading"}
      </span>
      {service.chunks !== null && <span>{service.chunks.toLocaleString()} chunks</span>}
    </p>
  );
}
