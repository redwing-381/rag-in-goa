# RAG in Goa

**Team McQuade** · Hacker House Goa 2026 shortlisting task

Voice in. Grounded answer out. Retrieval under **200 ms**, or we skip a stage
rather than miss the deadline. The model may only speak from retrieved
passages. If the corpus cannot support it, we refuse.

The index is English. You are not limited to English: Sarvam `saaras:v3`
(`mode=translate`) turns Hindi, Bengali, Tamil, or Marathi speech into an
English query. The answer comes back in the language you used.

```
speech ──► Sarvam STT (translate) ──► English query
                                           │
                                    input gate
                                           │
                    ONNX embed · HNSW · BM25 · RRF · top 3
                         (200 ms wall clock, rerank off)
                                           │
                                    domain gate
                                           │
                         OpenRouter (Cerebras → Groq)
                                           │
                                    output gate
                                           │
                              answer · sources · trace · TTS
```

- **Live API:** [https://rag-in-goa.fly.dev/health](https://rag-in-goa.fly.dev/health)
- **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Latency notes:** [`docs/LATENCY.md`](docs/LATENCY.md)
- **Deploy:** [`docs/DEPLOY.md`](docs/DEPLOY.md)

---

## Contents

- [Why this shape](#why-this-shape)
- [Stack](#stack)
- [Latency](#latency)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Try these questions](#try-these-questions)
- [API](#api)
- [Repository layout](#repository-layout)
- [Index](#index)
- [Deploy](#deploy)
- [Team](#team)
- [License](#license)

## Why this shape

MSMARCO-XI is translated MS MARCO QnA, not a generic web crawl. We built
pseudo-documents from the validation split (97,941 docs → 215,608 chunks)
and search them with hybrid retrieval.

Three decisions the demo depends on:

1. **Retrieval is ours. Generation is not.** Embed, HNSW, and BM25 run in
   process against a memory-mapped index. STT, TTS, and the LLM are remote
   and shown on their own spans. The 200 ms claim is retrieval only.
2. **One English index.** Cross-lingual input is a translation problem, not a
   second vector store.
3. **Refuse beats a fluent miss.** Input, domain, and output gates sit around
   the retriever and the LLM. A model “I don’t know” is a clean refusal, not
   a groundedness failure.

Rerank (MiniLM cross-encoder) is implemented and **off** by default. On this
laptop it was ~160 ms of a ~165 ms total. Turn it on with `RAGOA_RERANK=1`.

There is no opening or closing greeting. A question is a question.

## Stack

| Layer | Choice |
|---|---|
| Speech in | Sarvam `saaras:v3`, translate |
| Speech out | Sarvam Bulbul, speaker Ishita |
| Embed | `BAAI/bge-small-en-v1.5`, ONNX int8 at serve time |
| Dense | HNSW, 30 candidates |
| Lexical | BM25, 30 candidates |
| Fusion | Reciprocal rank fusion, top 3 |
| Chunking (shipped) | Script-aware, 1800 / 200 |
| LLM | OpenRouter `openai/gpt-oss-120b`, Cerebras then Groq |
| API | FastAPI, SSE (`/ask/stream`, `/ask/audio/stream`) |
| UI | Next.js 15, languages `en` `hi` `bn` `ta` `mr` |
| Host | Fly.io API (Singapore, 4 GB) · Vercel frontend |

Six chunkers live under `ragoa/chunking/` (fixed, script-aware,
sentence-window, hierarchical, metadata-aware, semantic). Only script-aware
is served. That choice is one call site in `ragoa/chunking/registry.py`.

## Latency

400 English MSMARCO-XI validation queries, 10 warmup, rerank off, 200 ms
budget. Local is in-process `retriever.retrieve()`. Live is
`trace.retrieval_ms` on Fly, not the HTTP wait.

| | P50 | P95 | P99 |
|---|---:|---:|---:|
| Local · MacBook Air M2 | **7.3 ms** | **11.1 ms** | 13.7 ms |
| Live · Fly `sin` shared-cpu-2x 4 GB | **47.5 ms** | **76.9 ms** | 88.8 ms |

Both **PASS**. Same index. Live is slower because of shared CPU, not a
different corpus. Live HTTP P50 is ~1.1 s: retrieval ~48 ms, LLM ~617 ms,
the rest is the hop.

```bash
python benchmark.py 400
python benchmark.py 400 --url https://rag-in-goa.fly.dev
```

## Quick start

Python **3.11** (not 3.13). Keys in `.env` (never commit it):

```bash
cp .env.example .env
# OPENROUTER_API_KEY=  https://openrouter.ai/keys
# SARVAM_API_KEY=      https://dashboard.sarvam.ai
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]"

# unpack the Colab index so data/index/ and data/onnx/ exist
set -a && source .env && set +a
RAGOA_INDEX_DIR=data/index RAGOA_ENCODER=onnx RAGOA_STT=1 \
  python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Wait for `ready: 215,608 chunks`. Then:

```bash
cd apps/web
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for answers |
| `SARVAM_API_KEY` | — | Required for speech |
| `RAGOA_INDEX_DIR` | `data/index` | Served index |
| `RAGOA_ENCODER` | `onnx` in Docker | `onnx` · `st` · `hash` |
| `RAGOA_RERANK` | `0` | Set `1` to enable the cross-encoder |
| `RAGOA_STT` | `1` | Set `0` in tests |
| `RAGOA_CORS_ORIGINS` | `*` | Tighten to the Vercel origin in production |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | FastAPI origin. Keys stay on the server |

## Try these questions

The corpus is MS MARCO, not the live web.

- what type of mountain is Mount Fuji
- who was Bridget Moynahan married to
- what is Kinsey most known for
- definition of philosophy
- என்ன வகையான மலை எம்டி ஃபுஜி? — set the UI language to Tamil

Out of domain (should refuse): weather in Panaji tomorrow.  
Unsafe (should refuse): how to make a bomb at home.

## API

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Ready flag, chunk count, warmup ms |
| `GET` | `/config` | Non-secret settings |
| `POST` | `/ask` | Text in, `AskResponse` out |
| `POST` | `/ask/stream` | Same turn as SSE (`stage`, `token`, `final`) |
| `POST` | `/ask/audio` | WAV in, grounded answer |
| `POST` | `/ask/audio/stream` | Speech, then the same SSE events |
| `POST` | `/speak` | Text → WAV (TTS) |

`AskResponse.trace` is the same object the UI and `benchmark.py` use.

## Repository layout

```
apps/api/          FastAPI
apps/web/          Next.js UI
ragoa/             retrieval, harness, guardrails, STT/TTS
bench/             extra latency harness
benchmark.py       judge table: embed / search / total
notebooks/         Colab index build
scripts/           data, index, ONNX export
docs/              architecture, latency, deploy
tests/
```

`data/index` and `data/onnx` are gitignored (~650 MB + ONNX graphs). They
are copied into the Docker image on this machine at `fly deploy`.

## Index

Validation split only (Hindi shard; English passages are identical across
languages). Rebuild on a GPU:

[`notebooks/build_index_colab.ipynb`](notebooks/build_index_colab.ipynb)

```bash
python -m ragoa.data.docbuilder
python scripts/build_index.py --encoder st
python scripts/export_onnx.py
```

## Deploy

The website on Vercel cannot mmap the index. Put **Next.js on Vercel** and
**FastAPI + index on Fly**.

```bash
fly auth login
fly launch --no-deploy --copy-config
fly secrets set OPENROUTER_API_KEY=... SARVAM_API_KEY=...
fly deploy
```

```bash
cd apps/web
npx vercel --prod
```

Set `NEXT_PUBLIC_API_URL=https://rag-in-goa.fly.dev` in Vercel and redeploy.

`fly.toml` uses `auto_stop_machines = 'stop'` and `min_machines_running = 0`.
After idle, the first request may take ~60 s or return 502 while the 4 GB
machine loads the index. Wake it with `/health`, or set
`min_machines_running = 1` (~$27/month always on in Singapore).

## Team

| GitHub |
|---|
| [redwing-381](https://github.com/redwing-381) |
| [MerlynNatty](https://github.com/MerlynNatty) |
| [shobanravichandran](https://github.com/shobanravichandran) |

## License

Code is **MIT**.

| Asset | Terms |
|---|---|
| [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) | Upstream MS MARCO + AI4Bharat |
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT |
| [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Optional reranker |
