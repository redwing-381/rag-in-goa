"""Convert a raw MSMARCO-XI validation shard into a slim, fast-to-read parquet.

Why this exists: each raw shard is a *single* 1.16 GB row group, so every read
of the raw file decodes the whole thing. We pay that cost once here and write a
copy with small row groups, after which downstream reads are effectively free.

Also reports corpus statistics, since the size of the deduplicated English
passage set determines how large the vector index will be.

    python scripts/prepare_data.py --lang hi
    python scripts/prepare_data.py --all --stats-only
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw")
SLIM_DIR = Path("data/slim")

# Raw shard filenames use 3-letter prefixes that do not always match ISO codes
# (the dataset README gets several of these wrong).
SHARDS = {
    "hi": "hinval",
    "bn": "benval",
    "ta": "tamval",
    "mr": "marval",
}

READ_COLUMNS = [
    "query_id",
    "query_type",
    "query",
    "Answer",
    "Eng_Query",
    "Eng_Answer",
    "passages",
]

SLIM_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("query_type", pa.string()),
        ("eng_query", pa.string()),
        ("translated_query", pa.string()),
        ("eng_answer", pa.string()),
        ("translated_answer", pa.string()),
        ("passages_eng", pa.list_(pa.string())),
        ("passages_translated", pa.list_(pa.string())),
        ("is_selected", pa.list_(pa.int8())),
    ]
)

ROW_GROUP_SIZE = 2000
BATCH_SIZE = 500


def clean_query(text: str | None) -> str:
    """Strip the leading '. ' artifact present on many Eng_Query values."""
    if not text:
        return ""
    return text.lstrip(". ").strip()


def passage_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


class Stats:
    def __init__(self) -> None:
        self.rows = 0
        self.passages = 0
        self.unique_passages: set[bytes] = set()
        self.doc_chars = 0
        self.min_doc_chars = 10**9
        self.max_doc_chars = 0
        self.n_selected_hist: dict[int, int] = {}
        self.query_types: dict[str, int] = {}
        self.empty_answers = 0

    def observe(self, row: dict) -> None:
        self.rows += 1
        eng = row["passages"]["English_passages"] or []
        selected = row["passages"]["is_selected"] or []

        self.passages += len(eng)
        for text in eng:
            self.unique_passages.add(passage_hash(text))

        doc_len = sum(len(p) for p in eng)
        self.doc_chars += doc_len
        self.min_doc_chars = min(self.min_doc_chars, doc_len)
        self.max_doc_chars = max(self.max_doc_chars, doc_len)

        n_sel = sum(1 for s in selected if s)
        self.n_selected_hist[n_sel] = self.n_selected_hist.get(n_sel, 0) + 1

        qt = row["query_type"] or "UNKNOWN"
        self.query_types[qt] = self.query_types.get(qt, 0) + 1

        if not row.get("Eng_Answer"):
            self.empty_answers += 1

    def report(self, lang: str) -> None:
        uniq = len(self.unique_passages)
        avg_doc = self.doc_chars / max(self.rows, 1)
        print(f"\n===== corpus stats: {lang} =====")
        print(f"rows (queries)          : {self.rows:,}")
        print(f"passages (with dupes)   : {self.passages:,}")
        print(f"passages (deduped)      : {uniq:,}  ({100 * uniq / max(self.passages, 1):.1f}%)")
        print(f"avg passages per query  : {self.passages / max(self.rows, 1):.2f}")
        print(f"pseudo-doc chars        : avg {avg_doc:,.0f} / "
              f"min {self.min_doc_chars:,} / max {self.max_doc_chars:,}")
        print(f"approx tokens per doc   : {avg_doc / 4:,.0f}")
        print(f"queries w/o Eng_Answer  : {self.empty_answers:,} "
              f"({100 * self.empty_answers / max(self.rows, 1):.1f}%)")
        print(f"gold passages per query : {dict(sorted(self.n_selected_hist.items()))}")
        by_count = dict(sorted(self.query_types.items(), key=lambda kv: -kv[1]))
        print(f"query types             : {by_count}")

        answerable = self.rows - self.n_selected_hist.get(0, 0)
        print(f"answerable queries      : {answerable:,} (have >=1 gold passage)")

        # Index sizing at 384 dims, float32, for a few chunk-size scenarios.
        print("\nindex sizing (384-dim float32):")
        for label, n_vecs in (
            ("one vector per passage", uniq),
            ("one vector per pseudo-doc", self.rows),
            ("~450-token chunks (est.)", int(self.rows * max(avg_doc / 1800, 1))),
        ):
            mb = n_vecs * 384 * 4 / 1e6
            print(f"  {label:26s}: {n_vecs:>9,} vectors  {mb:>7,.0f} MB raw")


def convert(lang: str, limit: int | None, stats_only: bool) -> None:
    shard = SHARDS[lang]
    src = RAW_DIR / f"{shard}.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src}; download it first")

    SLIM_DIR.mkdir(parents=True, exist_ok=True)
    dst = SLIM_DIR / f"{lang}.parquet"

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    print(f"[{lang}] {src.name}: {total:,} rows, {pf.metadata.num_row_groups} row group(s), "
          f"{src.stat().st_size / 1e9:.2f} GB")

    stats = Stats()
    writer = None if stats_only else pq.ParquetWriter(dst, SLIM_SCHEMA, compression="zstd")
    t0 = time.perf_counter()
    written = 0

    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=READ_COLUMNS):
            rows = batch.to_pylist()
            out: dict[str, list] = {name: [] for name in SLIM_SCHEMA.names}

            for row in rows:
                if limit is not None and stats.rows >= limit:
                    break
                stats.observe(row)
                passages = row["passages"]
                out["query_id"].append(row["query_id"])
                out["query_type"].append(row["query_type"])
                out["eng_query"].append(clean_query(row["Eng_Query"]))
                out["translated_query"].append((row["query"] or "").strip())
                out["eng_answer"].append(row["Eng_Answer"] or "")
                out["translated_answer"].append(row["Answer"] or "")
                out["passages_eng"].append(passages["English_passages"] or [])
                out["passages_translated"].append(passages["Translated_passages"] or [])
                out["is_selected"].append([int(s) for s in (passages["is_selected"] or [])])

            if writer is not None and out["query_id"]:
                writer.write_table(
                    pa.Table.from_pydict(out, schema=SLIM_SCHEMA),
                    row_group_size=ROW_GROUP_SIZE,
                )
                written += len(out["query_id"])

            elapsed = time.perf_counter() - t0
            rate = stats.rows / max(elapsed, 1e-6)
            print(f"\r[{lang}] {stats.rows:,}/{total:,} rows  {rate:,.0f} rows/s  "
                  f"{elapsed:.0f}s", end="", flush=True)

            if limit is not None and stats.rows >= limit:
                break
    finally:
        if writer is not None:
            writer.close()

    print()
    stats.report(lang)
    if writer is not None:
        print(f"\nwrote {written:,} rows -> {dst} ({dst.stat().st_size / 1e6:,.0f} MB)")
    print(f"elapsed: {time.perf_counter() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(SHARDS), help="single language to convert")
    ap.add_argument("--all", action="store_true", help="convert every downloaded shard")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows (smoke test)")
    ap.add_argument("--stats-only", action="store_true", help="report stats without writing")
    args = ap.parse_args()

    langs = sorted(SHARDS) if args.all else ([args.lang] if args.lang else [])
    if not langs:
        raise SystemExit("pass --lang <code> or --all")

    for lang in langs:
        convert(lang, args.limit, args.stats_only)


if __name__ == "__main__":
    main()
