# Latency

The brief asks for P50 / P70 / P100 on the retrieval path (chunking is
offline; generation is a remote GPU and is reported separately). These
numbers are from `benchmark.py` against the **served** stack: the
Colab-built index, ONNX int8 embedder, BM25, and RRF fusion, with rerank
off and the same 200 ms deadline the API uses.

## Method

- **Index:** 215,608 script-aware chunks over 97,941 validation documents.
- **Queries:** 400 English MS MARCO-XI validation questions, after 10 warmup
  calls.
- **Local machine:** MacBook Air M2, 8 GB RAM, ONNX Runtime, 4 intra-op threads.
- **Live machine:** Fly.io Singapore, shared-cpu-2x, 4 GB. Live timings are
  `trace.retrieval_ms` from `POST /ask`, not HTTP round-trip.
- **What is timed:** embed → dense HNSW → BM25 → fusion → context expand.
  Speech-to-text and the LLM are not in this table.

## Results (rerank off)

Local, 19 Aug 2026:

```
stage                avg     p50     p95     p99   (ms)
embed               2.34    2.21    3.13    3.94
search              5.28    4.99    8.71   10.94
total               7.61    7.33   11.14   13.65
```

Live Fly Singapore, same 400 queries:

```
stage                avg     p50     p95     p99   (ms)
embed              20.47   18.61   33.49   41.24
search             29.00   25.21   55.41   62.96
total              49.47   47.50   76.85   88.80
```

Both PASS the 200 ms budget at P95. Live HTTP P50 is ~1.1 s because of the
LLM (~617 ms) plus the network hop; that is not the retrieval claim.

With rerank on (earlier local run): total P50 165 ms, P95 197 ms. Rerank
alone was ~160 ms, so it is off by default (`RAGOA_RERANK=0`).

Reproduce:

```bash
python benchmark.py 400
python benchmark.py 400 --url https://rag-in-goa.fly.dev
```
