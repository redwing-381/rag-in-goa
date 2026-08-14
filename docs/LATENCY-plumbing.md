# Latency

```
index            : 43,075 chunks from 20,000 docs, strategy=script_aware
encoder          : hash (hash)
                   WARNING: HashEncoder is not semantic. These are plumbing timings only, not retrieval quality.
lexical leg      : on
reranker         : off
queries measured : 200 (after 10 warmup)
budget           : 200 ms
cold first call  : 21.9 ms (excluded from percentiles)

stage                 p50      p70      p95      p99      p100
--------------------------------------------------------------
embed_query          0.02     0.02     0.03     0.03      0.04
dense_search         0.30     0.33     0.40     0.44      0.50
sparse_search        0.89     1.28     2.67     3.51      5.28
fusion               0.01     0.01     0.01     0.01      0.01
expand_context       0.01     0.01     0.02     0.05      0.07
--------------------------------------------------------------
TOTAL                1.96     2.64     4.32     5.11      6.23

under 200 ms: p50 yes, p95 yes, p100 yes  -> PASS
over budget      : 0/200 requests
degradations     : none (full pipeline)
```
