# RAG in Goa

Voice-enabled RAG over [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
for the HH Goa 2026 shortlisting task.

Speak (or type) a question in English, Hindi, Bengali, Tamil, or Marathi.
Sarvam transcribes it to English, hybrid retrieval finds supporting passages
in under 200 ms, and a harnessed LLM answers only from those passages — or
refuses. The first screen has tap-to-ask examples so you do not have to
invent a question the corpus can answer.

## Retrieval latency

Measured on the served stack (ONNX int8 embed + BM25 + ONNX int8 rerank),
400 validation queries after warmup, MacBook Air M2. Full table and method
in [`docs/LATENCY.md`](docs/LATENCY.md).

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| Retrieval | **174.5 ms** | **176.3 ms** | **358.2 ms** |

395 / 400 requests finished inside 200 ms. P100 is five rerank outliers.
Speech-to-text and generation are reported separately; they are remote APIs.

## What is in the box

- **STT:** Sarvam `saaras:v3`, translate mode (Indic speech → English query).
- **Chunking:** six strategies; shipped default is script-aware 1800 / 200.
- **Index:** 215,608 chunks, 97,941 validation documents, HNSW + BM25.
- **Harness:** OpenRouter structured JSON, retries, circuit breaker, ordinal citations.
  If the model writes “I do not have that information” but the retrieved
  passages actually mention the question, it is asked once more before we refuse.
- **Guardrails:** unsafe / injection / unintelligible input; out-of-domain;
  ungrounded or badly cited output. A model “I don’t know” is a clean refusal,
  not a groundedness failure on that sentence.
- **UI:** Next.js mic capture, sample questions, named wait stages, spoken
  answers when the browser has a voice, short sources, and a collapsed
  latency trace.

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Local run

```bash
cp .env.example .env          # OPENROUTER_API_KEY, SARVAM_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]"

# index from Colab: unpack ragoa-index.tar.gz so data/index/ exists
RAGOA_ENCODER=onnx RAGOA_STT=1 python -m uvicorn apps.api.main:app --port 8000

cd apps/web && npm install && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:3000. Rebuild the index on a GPU with
[`notebooks/build_index_colab.ipynb`](notebooks/build_index_colab.ipynb).

Questions that work well on this index:

- what type of mountain is Mount Fuji
- who was Bridget Moynahan married to
- what is Kinsey most known for
- definition of philosophy
- என்ன வகையான மலை எம்டி ஃபுஜி? (set the language to Tamil)

## Deploy

API (Fly.io), from a machine that already has `data/index` and `data/onnx`:

```bash
fly launch --no-deploy --copy-config
fly secrets set OPENROUTER_API_KEY=... SARVAM_API_KEY=...
fly deploy
```

Frontend (Vercel), with `NEXT_PUBLIC_API_URL` set to the Fly URL:

```bash
cd apps/web
npx vercel --prod
```

## Team

- [redwing-381](https://github.com/redwing-381)
- [MerlynNatty](https://github.com/MerlynNatty)
- [shobanravichandran](https://github.com/shobanravichandran)

## Licence

Code in this repository is MIT.

The corpus is [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
(MS MARCO passages and AI4Bharat translations). Use of the dataset follows
those upstream terms. Embeddings use
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT).
Reranking uses
[cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
