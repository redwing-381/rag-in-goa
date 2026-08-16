/**
 * Mirrors ragoa/schemas.py. Kept hand-written rather than generated because the
 * surface is small and a generator would be another thing to keep running.
 *
 * If these drift from the Pydantic models the UI renders blanks, so any field
 * added there needs adding here.
 */

export type Language = "en" | "hi" | "bn" | "ta" | "mr";

/** Must stay in step with RefusalReason in ragoa/schemas.py. */
export type RefusalReason =
  | "unsafe_input"
  | "prompt_injection"
  | "unintelligible_audio"
  | "out_of_domain"
  | "not_grounded"
  | "invalid_citations"
  | "provider_failure";

export interface Citation {
  chunk_id: string;
  doc_id: string;
  text: string;
  translated_text: string | null;
  score: number;
}

export interface Span {
  name: string;
  duration_ms: number;
  metadata: Record<string, string | number | boolean>;
}

export interface Trace {
  request_id: string;
  spans: Span[];
  /** Stages the deadline logic chose to skip, e.g. "rerank_skipped". */
  degradations: string[];
  tool_calls: string[];
  retrieval_ms: number;
  llm_ttft_ms: number | null;
  total_ms: number;
  cache_hit: boolean;
}

export interface GroundednessReport {
  grounded: boolean;
  score: number;
  unsupported_sentences: string[];
  invalid_citations: string[];
}

export interface HistoryTurn {
  role: "user" | "assistant";
  text: string;
}

export interface LiveStage {
  name: string;
  duration_ms: number;
}

export type StreamEvent =
  | { type: "transcript"; text: string }
  | { type: "stage"; name: string; duration_ms: number }
  | { type: "token"; text: string }
  | { type: "final"; response: AskResponse };

export interface AskResponse {
  answer: string;
  refused: boolean;
  refusal_reason: RefusalReason | null;
  citations: Citation[];
  confidence: number;
  transcript: string | null;
  answer_language: Language;
  groundedness: GroundednessReport | null;
  trace: Trace;
}

export interface HealthResponse {
  ready: boolean;
  chunks: number | null;
  docs: number | null;
  strategy: string | null;
  encoder: string | null;
  has_sparse: boolean | null;
  warmup_ms: number | null;
  retrieval_budget_ms: number;
  has_tts: boolean;
}

export const LANGUAGES: { code: Language; label: string; native: string }[] = [
  { code: "en", label: "English", native: "English" },
  { code: "hi", label: "Hindi", native: "हिन्दी" },
  { code: "bn", label: "Bengali", native: "বাংলা" },
  { code: "ta", label: "Tamil", native: "தமிழ்" },
  { code: "mr", label: "Marathi", native: "मराठी" },
];

/** Human-readable refusal copy. The API sends a reason code, not prose. */
export const REFUSAL_COPY: Record<RefusalReason, string> = {
  unsafe_input: "The question was refused as unsafe.",
  prompt_injection: "The question looked like an attempt to override the instructions.",
  unintelligible_audio: "The audio could not be transcribed.",
  out_of_domain: "Nothing in the indexed corpus is relevant to this question.",
  not_grounded: "A draft answer was not supported by the retrieved passages, so it was withheld.",
  invalid_citations: "The draft answer cited passages that were never retrieved.",
  provider_failure: "The language model was unreachable, so no answer was generated.",
};
