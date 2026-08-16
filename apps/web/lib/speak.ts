import { ApiError, speakAudio } from "./api";
import type { Language } from "./types";

const LANG_TAGS: Record<Language, string[]> = {
  en: ["en-US", "en-GB", "en-IN", "en-AU", "en"],
  hi: ["hi-IN", "hi"],
  bn: ["bn-IN", "bn-BD", "bn"],
  ta: ["ta-IN", "ta"],
  mr: ["mr-IN", "mr"],
};

const PREFER = /samantha|google us english|google uk english|karen|moira|daniel|lekha|enhanced|premium|neural|natural/i;
const AVOID = /compact|novelty|whisper|zarvox|trinoids|boing|bad news|good news|cellos|organ|bells|pipe organ/i;

type SpeakJob = {
  text: string;
  language: Language;
  audio: Promise<Blob | null> | null;
};

let sarvamEnabled = false;
const playJobs: SpeakJob[] = [];
let draining = false;
let currentAudio: HTMLAudioElement | null = null;
let currentUrl: string | null = null;
let onSpeakStart: (() => void) | null = null;

export function setSpeakListener(listener: (() => void) | null): void {
  onSpeakStart = listener;
}

export function setSarvamTts(enabled: boolean): void {
  sarvamEnabled = enabled;
}

export function hasSarvamTts(): boolean {
  return sarvamEnabled;
}

function voices(): SpeechSynthesisVoice[] {
  if (typeof window === "undefined" || !window.speechSynthesis) return [];
  return window.speechSynthesis.getVoices();
}

/** Warm the voice list; Chrome only fills it after `voiceschanged`. */
export function preloadVoices(): void {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.getVoices();
}

export function voiceFor(language: Language): SpeechSynthesisVoice | null {
  const list = voices();
  if (list.length === 0) return null;
  const tags = LANG_TAGS[language];
  const matches: SpeechSynthesisVoice[] = [];
  for (const tag of tags) {
    const prefix = tag.split("-")[0].toLowerCase();
    for (const voice of list) {
      const lang = voice.lang.toLowerCase();
      if (lang === tag.toLowerCase() || lang.startsWith(prefix)) {
        if (!matches.includes(voice)) matches.push(voice);
      }
    }
    if (matches.length > 0) break;
  }
  if (matches.length === 0) return null;

  const usable = matches.filter((voice) => !AVOID.test(voice.name));
  const pool = usable.length > 0 ? usable : matches;
  const preferred = pool.find((voice) => PREFER.test(voice.name));
  if (preferred) return preferred;
  const local = pool.find((voice) => voice.localService);
  return local ?? pool[0];
}

export function canSpeak(language: Language): boolean {
  return sarvamEnabled || Boolean(voiceFor(language));
}

/** Drop citation markers so the voice does not say “bracket one”. */
export function cleanForSpeech(text: string): string {
  return text
    .replace(/【\d+】/g, " ")
    .replace(/\[\s*"?\d+"?\s*(?:,\s*"?\d+"?\s*)*\]/g, " ")
    .replace(/\[\d+\]/g, " ")
    .replace(/\(\s*\d+\s*\)/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function enqueueBrowser(text: string, language: Language): boolean {
  if (typeof window === "undefined" || !window.speechSynthesis) return false;
  const voice = voiceFor(language);
  if (!voice) return false;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  utterance.rate = 0.92;
  utterance.pitch = 0.98;
  window.speechSynthesis.speak(utterance);
  return true;
}

function playBlob(blob: Blob): Promise<void> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = audio;
    currentUrl = url;
    onSpeakStart?.();
    const finish = () => {
      if (currentUrl === url) {
        currentAudio = null;
        currentUrl = null;
      }
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onended = finish;
    audio.onerror = finish;
    void audio.play().catch(finish);
  });
}

function startFetch(text: string, language: Language): Promise<Blob | null> | null {
  if (!sarvamEnabled) return null;
  return speakAudio(text, language).catch((err) => {
    if (err instanceof ApiError && err.status === 503) sarvamEnabled = false;
    return null;
  });
}

async function drainQueue(): Promise<void> {
  if (draining) return;
  draining = true;
  while (playJobs.length > 0) {
    const job = playJobs.shift();
    if (!job) break;
    const blob = job.audio ? await job.audio : null;
    if (blob) {
      await playBlob(blob);
      continue;
    }
    enqueueBrowser(job.text, job.language);
    onSpeakStart?.();
    await whenBrowserSpeechEnds();
  }
  draining = false;
}

async function whenBrowserSpeechEnds(): Promise<void> {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  await new Promise<void>((resolve) => {
    const check = () => {
      if (!window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
        resolve();
        return;
      }
      window.setTimeout(check, 80);
    };
    check();
  });
}

/** Speak `text` in `language`. Cancels anything already playing. */
export function speak(text: string, language: Language): boolean {
  stopSpeaking();
  return speakNext(text, language);
}

/** Queue a phrase. Sarvam first; browser voice if that path is down. */
export function speakNext(text: string, language: Language): boolean {
  const spoken = cleanForSpeech(text);
  if (!spoken) return false;
  if (!sarvamEnabled && !voiceFor(language)) return false;
  playJobs.push({ text: spoken, language, audio: startFetch(spoken, language) });
  void drainQueue();
  return true;
}

export function stopSpeaking(): void {
  playJobs.length = 0;
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (currentUrl) {
    URL.revokeObjectURL(currentUrl);
    currentUrl = null;
  }
  if (typeof window !== "undefined" && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

/** Resolves once both the Sarvam queue and the browser voice are idle. */
export function whenSpeechEnds(): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      const browserBusy =
        typeof window !== "undefined" &&
        Boolean(window.speechSynthesis?.speaking || window.speechSynthesis?.pending);
      if (!draining && playJobs.length === 0 && !currentAudio && !browserBusy) {
        resolve();
        return;
      }
      window.setTimeout(check, 80);
    };
    check();
  });
}

const SENTENCE_END = /[.!?।॥](?:["')\]]+)?(?:\s+|$)/;

/** Pull finished sentences off a streaming buffer. Remainder stays unsaid. */
export function takeSentences(buffer: string): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  const pattern = new RegExp(SENTENCE_END, "g");
  while ((match = pattern.exec(buffer)) !== null) {
    const end = match.index + match[0].length;
    const piece = buffer.slice(cursor, end).trim();
    if (piece) sentences.push(piece);
    cursor = end;
  }
  return { sentences, rest: buffer.slice(cursor) };
}

/** Speak each finished sentence as soon as it lands, so TTS can start early. */
export function takeSpeakable(buffer: string): { chunks: string[]; rest: string } {
  const { sentences, rest } = takeSentences(buffer);
  return { chunks: sentences.filter((sentence) => sentence.length >= 8), rest };
}
