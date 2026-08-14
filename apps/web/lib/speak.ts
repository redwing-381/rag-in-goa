import type { Language } from "./types";

const LANG_TAGS: Record<Language, string[]> = {
  en: ["en-US", "en-GB", "en"],
  hi: ["hi-IN", "hi"],
  bn: ["bn-IN", "bn-BD", "bn"],
  ta: ["ta-IN", "ta"],
  mr: ["mr-IN", "mr"],
};

function voices(): SpeechSynthesisVoice[] {
  if (typeof window === "undefined" || !window.speechSynthesis) return [];
  return window.speechSynthesis.getVoices();
}

export function voiceFor(language: Language): SpeechSynthesisVoice | null {
  const list = voices();
  if (list.length === 0) return null;
  const tags = LANG_TAGS[language];
  for (const tag of tags) {
    const exact = list.find((voice) => voice.lang.toLowerCase() === tag.toLowerCase());
    if (exact) return exact;
    const prefix = tag.split("-")[0].toLowerCase();
    const loose = list.find((voice) => voice.lang.toLowerCase().startsWith(prefix));
    if (loose) return loose;
  }
  return null;
}

/** Speak `text` in `language`. No-ops if the browser has no matching voice. */
export function speak(text: string, language: Language): boolean {
  if (typeof window === "undefined" || !window.speechSynthesis) return false;
  const voice = voiceFor(language);
  if (!voice) return false;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  window.speechSynthesis.speak(utterance);
  return true;
}

export function stopSpeaking(): void {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
}
