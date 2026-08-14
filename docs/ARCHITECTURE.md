# Architecture

Voice in, grounded answer out. Retrieval is ours and is held under a
200 ms deadline. Generation is someone else's GPU and is timed on its
own span so we never fold a network call into a number we cannot keep.

```
speech ──► Sarvam saaras:v3 (translate) ──► English query
                                              │
                                              ▼
                                    input gate (unsafe / injection / noise)
                                              │
                    ┌──────────── embed (ONNX int8 bge-small) ────────────┐
                    │  dense HNSW          BM25          RRF fusion       │
                    │         └── deadline ──► rerank (ONNX int8) ──► k=3 │
                    └─────────────────────────────────────────────────────┘
                                              │
                                    domain gate (retrieval score)
                                              │
                         OpenRouter (Cerebras → Groq) structured JSON
                                              │
                         output gate (citations + groundedness)
                                              │
                                    answer + sources + trace
```

## Chunking

MS MARCO ships ~60-word passages. Indexing those as-is would make every
strategy look the same. We concatenate each query's candidates into a
pseudo-document (~2.6 KB) and record gold-answer character spans, then
split with a real chunker.

Six strategies live under `ragoa/chunking/`:

| Strategy | What it does |
|---|---|
| Fixed | Character windows, overlap sweep |
| Script-aware | Recursive split on Indic danda, punctuation, then length |
| Sentence-window | Small unit for search, wider window for the LLM |
| Hierarchical | Child chunks for index, parent for generation |
| Metadata-aware | Query-type / title prefix on a base chunker |
| Semantic | Split on embedding distance between sentences |

The shipped default is **script-aware, 1800 characters, 200 overlap**
(`ragoa/chunking/registry.py`). That is one call site, so the served
index and the notebook cannot drift.

## Retrieval

Hybrid search with a hard wall-clock budget (`ragoa/index/retriever.py`):

1. Embed the query (ONNX int8).
2. HNSW cosine search, 30 candidates.
3. BM25, 30 candidates, if reserve remains.
4. Reciprocal rank fusion.
5. Cross-encoder rerank of 12, if reserve remains.
6. Return top 3, optionally expanded to a parent/window span.

Reserves are set from measured P95 stage costs, not guesses. If a stage
would blow the 200 ms budget it is skipped and listed on `trace.degradations`.

## Harness

`ragoa/harness/` is not a raw prompt call:

- JSON schema with `require_parameters: true` so OpenRouter cannot silently
  drop structured output.
- Provider order Cerebras → Groq on one model.
- Retries with jitter, circuit breaker, named `provider_failure` if both die.
- The model cites passage ordinals `[1]`, `[2]`; we map those back to chunk
  ids so a mistyped opaque id cannot refuse a correct answer.

## Guardrails

| Gate | When | What it refuses |
|---|---|---|
| Input | Before retrieval | Unsafe how-tos, prompt injection, empty or garbled transcripts |
| Domain | After retrieval | Best score below threshold (nothing in the corpus is close) |
| Output | After the LLM | Invented citations, answers the passages do not support |

A model that writes “I cannot help with that” is reported as `unsafe_input`,
not `not_grounded`. Low-confidence extractive fallbacks refuse rather than
quote a random passage.

## Voice

Sarvam `saaras:v3` runs in `mode=translate`. Indic speech becomes an
English query for the English-only index. `lang` stays the language the
user spoke and drives the answer language.

## What is not in the 200 ms number

STT and the LLM are timed and shown in the UI. They are not part of the
P50 / P70 / P100 table in `docs/LATENCY.md`. Claiming them would be
claiming someone else's SLO.
