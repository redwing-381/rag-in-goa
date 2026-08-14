# Latency

The brief asks for P50 / P70 / P100 on the retrieval path (chunking is
offline; generation is a remote GPU and is reported separately). These
numbers are from `bench/latency.py` against the **served** stack: the
Colab-built index, ONNX int8 embedder, BM25, RRF fusion, and ONNX int8
reranker, with the same 200 ms deadline the API uses.

## Method

- **Index:** 215,608 script-aware chunks over 97,941 validation documents.
- **Queries:** 400 English MS MARCO-XI validation questions, after 10 warmup
  calls. The first (cold) call is printed and excluded from the percentiles.
- **Machine:** MacBook Air M2, 8 GB RAM, ONNX Runtime, 4 intra-op threads.
- **What is timed:** embed → dense HNSW → BM25 → fusion → rerank → context
  expand. Speech-to-text and the LLM are not in this table; they are
  network calls and show up on their own spans in the UI.
- **P100** is a single worst request. We also report P95 / P99 so one
  page-fault does not define the system.

## Results

```
index            : 215,608 chunks from 97,941 docs, strategy=script_aware
encoder          : onnx (BAAI/bge-small-en-v1.5)
lexical leg      : on
reranker         : cross-encoder/ms-marco-MiniLM-L-6-v2
queries measured : 400 (after 10 warmup)
budget           : 200 ms
cold first call  : 218.7 ms (excluded from percentiles)

stage                 p50      p70      p95      p99      p100
--------------------------------------------------------------
embed_query          4.40     4.77     5.85     7.58     16.34
dense_search         1.67     2.16     5.22     9.97     16.34
sparse_search        4.11     4.91     7.68     9.83     14.81
fusion               0.02     0.03     0.04     0.06      0.15
rerank             161.51   163.59   169.28   190.18    344.02
expand_context       0.06     0.08     0.12     0.26      3.26
--------------------------------------------------------------
TOTAL              174.51   176.33   184.28   206.06    358.20

under 200 ms: p50 yes, p95 yes, p100 no  -> PASS at p95
over budget      : 5/400 requests
degradations     : none (full pipeline)
```

| | P50 | P70 | P100 |
|---|---:|---:|---:|
| Retrieval (this bench) | **174.5 ms** | **176.3 ms** | **358.2 ms** |

395 / 400 requests finished inside 200 ms. The five misses are rerank
outliers (P100 rerank 344 ms). The deadline can skip rerank or BM25 when
the reserve is gone; it did not fire on this run because those five still
started rerank with budget left, then overran.

Reproduce:

```bash
python bench/latency.py --index-dir data/index --encoder onnx --queries 400
```
