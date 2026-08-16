import type { Citation, Language, Trace } from "./types";

export type SessionTurn = {
  id: string;
  query: string;
  transcript: string | null;
  answer: string;
  lang: Language;
  trace: Trace | null;
  refused: boolean;
  citations: Citation[];
};

const KEY = "ragoa.session.turns";

export function loadTurns(): SessionTurn[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as SessionTurn[];
    if (!Array.isArray(parsed)) return [];
    return parsed.map((turn) => ({
      ...turn,
      citations: turn.citations ?? [],
    }));
  } catch {
    return [];
  }
}

export function saveTurns(turns: SessionTurn[]): void {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(KEY, JSON.stringify(turns));
}

export function newTurnId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
