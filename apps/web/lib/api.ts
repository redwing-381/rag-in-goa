/**
 * Typed client for the FastAPI service.
 *
 * The API URL is read at build time from NEXT_PUBLIC_API_URL. No key ever reaches
 * the browser: Sarvam and OpenRouter are called server-side, and the browser only
 * ever talks to our own service.
 */

import type { AskResponse, HealthResponse, Language } from "./types";

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** FastAPI reports errors as {detail: ...}; surface that rather than a bare status. */
async function readError(response: Response): Promise<never> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail)) detail = JSON.stringify(body.detail);
  } catch {
    // Non-JSON error body; the status text is the best we have.
  }
  throw new ApiError(detail, response.status);
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${BASE}/health`, { cache: "no-store" });
  if (!response.ok) await readError(response);
  return response.json();
}

export async function askText(
  query: string,
  lang: Language,
  topK = 3,
): Promise<AskResponse> {
  const response = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, lang, top_k: topK }),
  });
  if (!response.ok) await readError(response);
  return response.json();
}

export async function askAudio(
  audio: Blob,
  lang: Language,
  topK = 3,
): Promise<AskResponse> {
  const form = new FormData();
  form.append("file", audio, "question.wav");
  form.append("lang", lang);
  form.append("top_k", String(topK));

  const response = await fetch(`${BASE}/ask/audio`, { method: "POST", body: form });
  if (!response.ok) await readError(response);
  return response.json();
}

export { BASE as API_BASE };
