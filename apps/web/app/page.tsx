"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type OrbState } from "@/components/AgentOrb";
import { Composer } from "@/components/Composer";
import { HeroCard } from "@/components/HeroCard";
import { ChevronIcon, MenuIcon, SlidersIcon } from "@/components/icons";
import { InfoSheet } from "@/components/InfoSheet";
import { LiveStages } from "@/components/LiveStages";
import { useRecordingClock } from "@/components/MicButton";
import { SampleQuestions } from "@/components/SampleQuestions";
import { SidebarHistory } from "@/components/SidebarHistory";
import { TranscriptPane } from "@/components/TranscriptPane";
import { TracePanel } from "@/components/TracePanel";
import { ApiError, askAudioStream, askStream, health } from "@/lib/api";
import { MicRecorder, RecorderError } from "@/lib/recorder";
import { isEndCommand } from "@/lib/greetings";
import { loadTurns, newTurnId, saveTurns, type SessionTurn } from "@/lib/session";
import {
  preloadVoices,
  setSarvamTts,
  speak,
  stopSpeaking,
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
  const [waitHint, setWaitHint] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [infoId, setInfoId] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const recorder = useRef<MicRecorder | null>(null);
  const vad = useRef<ReturnType<typeof createVad> | null>(null);
  const heardSpeech = useRef(false);
  const sending = useRef(false);
  const ending = useRef(false);
  const cycle = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const playingIdRef = useRef<string | null>(null);
  const languageRef = useRef(language);
  const orbRef = useRef(orb);
  languageRef.current = language;
  orbRef.current = orb;

  useEffect(() => {
    preloadVoices();
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

  const endConversation = useCallback(async () => {
    if (ending.current) return;
    ending.current = true;
    cycle.current += 1;
    const id = cycle.current;
    haltCapture();
    stopSpeaking();
    if (id !== cycle.current) {
      ending.current = false;
      return;
    }
    setOrb("idle");
    ending.current = false;
  }, [haltCapture]);

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
      if (!fromAudio) setDraftYou(queryHint);

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
        setDraftAgent(last.answer);
        setDraftRefused(true);
      } else {
        setDraftAgent(last.answer);
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
      setOrb("idle");
    },
    [endConversation, pushUserLine],
  );

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
    stopSpeaking();
    playingIdRef.current = null;
    setPlayingId(null);
    sending.current = false;
    heardSpeech.current = false;
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
          minSpeechMs: 700,
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
  }, [service?.ready, stopAndSend]);

  const seconds = useRecordingClock(orb === "listening", () => void stopAndSend());

  useEffect(() => {
    if (orb === "listening" && seconds >= MAX_SECONDS) void stopAndSend();
  }, [orb, seconds, stopAndSend]);

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
      setTyped("");
      setError(null);
      stopSpeaking();
      if (isEndCommand(text)) {
        pushUserLine(text);
        await endConversation();
        return;
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
    [consumeStream, endConversation, historyForRequest, pushUserLine],
  );

  const playAnswer = useCallback((id: string, text: string, lang: Language) => {
    if (playingIdRef.current === id) {
      stopSpeaking();
      playingIdRef.current = null;
      setPlayingId(null);
      return;
    }
    stopSpeaking();
    playingIdRef.current = id;
    setPlayingId(id);
    speak(text, lang);
    void whenSpeechEnds().then(() => {
      if (playingIdRef.current === id) {
        playingIdRef.current = null;
        setPlayingId(null);
      }
    });
  }, []);

  const endVoiceChat = useCallback(() => {
    cycle.current += 1;
    abort.current?.abort();
    sending.current = false;
    stopCapture();
    recorder.current?.cancel();
    recorder.current = null;
    stopSpeaking();
    setWaitHint(null);
    setStreaming(false);
    setOrb("idle");
  }, [stopCapture]);

  const newChat = useCallback(() => {
    ending.current = true;
    cycle.current += 1;
    haltCapture();
    stopSpeaking();
    playingIdRef.current = null;
    setPlayingId(null);
    ending.current = false;
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
    setMenuOpen(false);
    setSettingsOpen(false);
  }, [haltCapture]);

  const live = orb !== "idle" || streaming;
  const inFlight = streaming || Boolean(waitHint);
  const selected = turns.find((turn) => turn.id === selectedId) ?? null;
  const infoTurn = turns.find((turn) => turn.id === infoId) ?? null;
  const refused = live ? draftRefused : Boolean(selected?.refused ?? draftRefused);
  const shownTrace = live ? finalResponse?.trace : (selected?.trace ?? finalResponse?.trace);

  const corpusLine = service?.docs
    ? `MS MARCO-XI · ${service.docs.toLocaleString()} documents`
    : "MS MARCO-XI";

  const qaTurns = turns.filter((turn) => (turn.kind ?? "qa") === "qa");
  const lastTurn = qaTurns.at(-1) ?? null;
  const lastPlayable = lastTurn && lastTurn.answer && !lastTurn.refused ? lastTurn : null;

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-paper text-ink">
      <header className="shrink-0 px-3 py-2.5 pt-[max(0.625rem,env(safe-area-inset-top))] sm:px-5 lg:px-6 lg:py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <button
              type="button"
              onClick={() => setMenuOpen(true)}
              aria-label="Open contents"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-ink lg:hidden"
            >
              <MenuIcon />
            </button>
            <div className="min-w-0">
              <p className="truncate font-sans text-[17px] font-semibold leading-none tracking-tight sm:text-lg">
                RAG in Goa
              </p>
              <p className="mt-1 truncate text-[11px] tracking-wide text-muted">
                {corpusLine}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2 sm:gap-3">
            <ServiceStatus service={service} error={serviceError} />
            <label className="relative flex min-h-9 items-center rounded-full border border-rule bg-card px-3 text-sm">
              <span className="sr-only">Language</span>
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value as Language)}
                className="max-w-[7.25rem] appearance-none bg-transparent py-1.5 pr-4 text-sm text-ink focus:outline-none sm:max-w-none"
              >
                {LANGUAGES.map((entry) => (
                  <option key={entry.code} value={entry.code}>
                    {entry.native}
                  </option>
                ))}
              </select>
              <ChevronIcon className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rotate-90 text-muted" />
            </label>
            <div className="relative">
              <button
                type="button"
                onClick={() => setSettingsOpen((open) => !open)}
                aria-label="Session settings"
                className="flex h-9 w-9 items-center justify-center rounded-full border border-rule bg-card text-ink"
              >
                <SlidersIcon />
              </button>
              {settingsOpen && (
                <>
                  <button
                    type="button"
                    className="fixed inset-0 z-20"
                    aria-label="Close settings"
                    onClick={() => setSettingsOpen(false)}
                  />
                  <div className="absolute right-0 top-full z-30 mt-2 w-44 rounded-2xl border border-rule bg-card p-1.5 shadow-[0_12px_30px_-18px_rgba(28,25,21,0.5)]">
                    <button
                      type="button"
                      onClick={newChat}
                      className="w-full rounded-xl px-3 py-2.5 text-left text-sm text-ink hover:bg-raised"
                    >
                      Clear session
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="hidden lg:block">
          <SidebarHistory
            turns={qaTurns}
            selectedId={selectedId}
            onSelect={(id) => {
              if (live) return;
              setSelectedId(id);
            }}
            onNewChat={newChat}
          />
        </div>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <TranscriptPane
            turns={turns}
            liveYou={inFlight ? draftYou : ""}
            liveAgent={inFlight ? draftAgent : ""}
            liveStatus={
              inFlight && !draftAgent
                ? waitHint || (draftYou ? "Looking that up…" : null)
                : inFlight && draftAgent && orb === "thinking"
                  ? "Writing the answer…"
                  : null
            }
            language={selected?.lang ?? language}
            refused={refused}
            live={inFlight}
            playingId={playingId}
            onListen={playAnswer}
            onInfo={setInfoId}
            emptyExtra={
              turns.length === 0 && !inFlight ? (
                <>
                  <HeroCard />
                  {(!service?.ready || serviceError) && (
                    <p className="mt-4 px-1 text-sm leading-relaxed text-accent">
                      If this page is still checking the API, or nothing works,
                      refresh the browser. The service may still be waking up.
                    </p>
                  )}
                  <SampleQuestions
                    disabled={!service?.ready}
                    onPick={(text, lang) => void submitText(text, lang)}
                  />
                </>
              ) : undefined
            }
          />
          <footer className="shrink-0 px-4 py-3 pb-[max(0.85rem,env(safe-area-inset-bottom))] sm:px-6 lg:px-8">
            <Composer
              typed={typed}
              onTyped={setTyped}
              onSubmit={() => void submitText(typed, language)}
              placeholder={
                orb === "idle"
                  ? "Or type the question here..."
                  : "Type is paused while the voice turn is running"
              }
              disabled={orb !== "idle" || !service?.ready}
              orb={orb}
              level={level}
              seconds={seconds}
              canPlay={Boolean(lastPlayable)}
              playing={Boolean(lastPlayable && playingId === lastPlayable.id)}
              onPlay={() => {
                if (!lastPlayable) return;
                playAnswer(lastPlayable.id, lastPlayable.answer, lastPlayable.lang);
              }}
              onMic={() => {
                if (orb === "listening") void stopAndSend();
                else void startListening();
              }}
              onEndChat={endVoiceChat}
              error={error}
              hintAction={
                lastTurn
                  ? {
                      label: "Sources",
                      onClick: () => setInfoId(lastTurn.id),
                    }
                  : undefined
              }
            />
          </footer>
        </div>

        <aside className="hidden h-full w-[300px] shrink-0 flex-col overflow-y-auto border-l border-rule bg-card/40 px-4 py-4 lg:flex">
          <p className="mb-3 font-sans text-[15px] text-ink">Timing</p>
          <LiveStages
            stages={stages}
            pending={streaming}
            audio={audioTurn}
            retrievalMs={shownTrace?.retrieval_ms}
          />
          {shownTrace ? (
            <div className="mt-4">
              <TracePanel
                trace={shownTrace}
                budgetMs={service?.retrieval_budget_ms ?? 200}
              />
            </div>
          ) : (
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Timings for the last turn land here.
            </p>
          )}
        </aside>
      </div>

      {menuOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-ink/35"
            aria-label="Close contents"
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-[min(20rem,86vw)] bg-paper pt-[env(safe-area-inset-top)] shadow-[8px_0_30px_-18px_rgba(28,25,21,0.45)]">
            <SidebarHistory
              className="flex h-full w-full flex-col"
              turns={qaTurns}
              selectedId={selectedId}
              onSelect={(id) => {
                if (live) return;
                setSelectedId(id);
                setMenuOpen(false);
              }}
              onNewChat={() => {
                newChat();
                setMenuOpen(false);
              }}
            />
          </div>
        </div>
      )}

      <InfoSheet
        open={Boolean(infoTurn)}
        onClose={() => setInfoId(null)}
        citations={infoTurn?.citations ?? []}
        trace={infoTurn?.trace ?? null}
        budgetMs={service?.retrieval_budget_ms ?? 200}
      />
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
      <p className="inline-flex max-w-[9rem] items-center gap-2 font-mono text-[11px] leading-snug text-accent sm:max-w-xs">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
        <span className="sm:hidden">Refresh</span>
        <span className="hidden sm:inline">Cannot reach the API. Refresh the page.</span>
      </p>
    );
  }
  if (!service) {
    return (
      <p className="font-mono text-[11px] text-muted">
        <span className="sm:hidden">…</span>
        <span className="hidden sm:inline">checking the service… refresh if this stays</span>
      </p>
    );
  }
  if (!service.ready) {
    return (
      <p className="font-mono text-[11px] text-muted">
        loading… refresh if it does not become ready
      </p>
    );
  }
  return (
    <p className="inline-flex items-center gap-1.5 font-mono text-[11px] text-muted">
      <span className="h-1.5 w-1.5 rounded-full bg-mint" />
      ready
    </p>
  );
}
