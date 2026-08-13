"""Turn MSMARCO-XI rows into pseudo-documents with gold spans.

MS MARCO ships passages pre-chunked at ~60 words, which would make chunking a
no-op variable: any strategy would just recover the passages it was given. So we
concatenate each query's ~10 candidate passages into one topical pseudo-document
(avg 3,156 chars / ~789 tokens) and record the character span of every
`is_selected` passage inside it.

Those spans are the ground truth: a retrieved chunk is a hit if it overlaps a
gold span. That is what makes the chunking comparison measurable rather than
rhetorical, and it is stated plainly in the README because it is a deliberate
construction, not an accident of the data.

Passages are joined with a single space on purpose. Joining with "\\n\\n" would
leave paragraph markers that a recursive splitter could follow to perfectly
recover the original passages, which would flatter every strategy equally and
tell us nothing.

The English side of every language shard is identical, so one shard is enough to
build the English corpus. Other shards contribute only translated text.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ragoa.schemas import Language, PseudoDocument, QueryType

SLIM_DIR = Path("data/slim")
CORPUS_DIR = Path("data/corpus")

JOINER = " "
NO_ANSWER = "No Answer Present."

DOCS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("query_id", pa.int64()),
        ("query_type", pa.string()),
        ("eng_query", pa.string()),
        ("eng_answer", pa.string()),
        ("text", pa.string()),
        ("answerable", pa.bool_()),
        ("gold_starts", pa.list_(pa.int32())),
        ("gold_ends", pa.list_(pa.int32())),
        ("passage_starts", pa.list_(pa.int32())),
        ("passage_ends", pa.list_(pa.int32())),
    ]
)

QUERIES_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("lang", pa.string()),
        ("query_type", pa.string()),
        ("eng_query", pa.string()),
        ("translated_query", pa.string()),
        ("eng_answer", pa.string()),
        ("translated_answer", pa.string()),
        ("answerable", pa.bool_()),
    ]
)


Span = tuple[int, int]


def assemble(passages: list[str], is_selected: list[int]) -> tuple[str, list[Span], list[Span]]:
    """Concatenate passages, returning the text plus passage and gold spans."""
    parts: list[str] = []
    passage_spans: list[tuple[int, int]] = []
    gold_spans: list[tuple[int, int]] = []
    cursor = 0

    for idx, passage in enumerate(passages):
        text = (passage or "").strip()
        if not text:
            continue
        if parts:
            cursor += len(JOINER)
            parts.append(JOINER)
        start = cursor
        end = start + len(text)
        parts.append(text)
        cursor = end

        passage_spans.append((start, end))
        if idx < len(is_selected) and is_selected[idx]:
            gold_spans.append((start, end))

    return "".join(parts), gold_spans, passage_spans


def iter_documents(lang: str = "hi", limit: int | None = None) -> Iterator[PseudoDocument]:
    """Yield English pseudo-documents built from one language shard."""
    src = SLIM_DIR / f"{lang}.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src}; run scripts/prepare_data.py --lang {lang}")

    pf = pq.ParquetFile(src)
    seen = 0
    columns = ["query_id", "query_type", "eng_query", "eng_answer",
               "translated_query", "translated_answer", "passages_eng", "is_selected"]

    for batch in pf.iter_batches(batch_size=500, columns=columns):
        for row in batch.to_pylist():
            text, gold, passage_spans = assemble(row["passages_eng"], row["is_selected"])
            if not text:
                continue

            yield PseudoDocument(
                doc_id=str(row["query_id"]),
                query_id=row["query_id"],
                query_type=QueryType(row["query_type"]),
                lang=Language.EN,
                text=text,
                gold_spans=gold,
                passage_offsets=passage_spans,
                eng_query=row["eng_query"],
                translated_query=row["translated_query"],
                eng_answer=row["eng_answer"],
                translated_answer=row["translated_answer"],
            )
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_docs(source_lang: str, limit: int | None) -> Path:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dst = CORPUS_DIR / "docs.parquet"
    writer = pq.ParquetWriter(dst, DOCS_SCHEMA, compression="zstd")

    buffer: dict[str, list] = {name: [] for name in DOCS_SCHEMA.names}
    n_docs = n_answerable = n_gold = 0
    total_chars = 0

    def flush() -> None:
        if buffer["doc_id"]:
            writer.write_table(pa.Table.from_pydict(buffer, schema=DOCS_SCHEMA),
                               row_group_size=1000)
            for value in buffer.values():
                value.clear()

    try:
        for doc in iter_documents(source_lang, limit):
            answerable = bool(doc.gold_spans)
            buffer["doc_id"].append(doc.doc_id)
            buffer["query_id"].append(doc.query_id)
            buffer["query_type"].append(doc.query_type.value)
            buffer["eng_query"].append(doc.eng_query)
            buffer["eng_answer"].append(doc.eng_answer or "")
            buffer["text"].append(doc.text)
            buffer["answerable"].append(answerable)
            buffer["gold_starts"].append([s for s, _ in doc.gold_spans])
            buffer["gold_ends"].append([e for _, e in doc.gold_spans])
            buffer["passage_starts"].append([s for s, _ in doc.passage_offsets])
            buffer["passage_ends"].append([e for _, e in doc.passage_offsets])

            n_docs += 1
            n_answerable += answerable
            n_gold += len(doc.gold_spans)
            total_chars += len(doc.text)

            if len(buffer["doc_id"]) >= 1000:
                flush()
        flush()
    finally:
        writer.close()

    print(f"docs      : {n_docs:,} pseudo-documents -> {dst} ({dst.stat().st_size / 1e6:,.0f} MB)")
    print(f"answerable: {n_answerable:,} ({100 * n_answerable / max(n_docs, 1):.1f}%), "
          f"{n_gold:,} gold spans total")
    print(f"unanswerable (labelled 'No Answer Present.'): "
          f"{n_docs - n_answerable:,} -> free ground truth for guardrail eval")
    print(f"avg doc   : {total_chars / max(n_docs, 1):,.0f} chars "
          f"(~{total_chars / max(n_docs, 1) / 4:,.0f} tokens)")
    return dst


def build_queries(langs: list[str], limit: int | None) -> Path:
    """Per-language query/answer table. Small, and it is our multilingual eval set."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dst = CORPUS_DIR / "queries.parquet"
    writer = pq.ParquetWriter(dst, QUERIES_SCHEMA, compression="zstd")
    counts: dict[str, int] = {}

    try:
        for lang in langs:
            src = SLIM_DIR / f"{lang}.parquet"
            if not src.exists():
                print(f"  skip {lang}: {src} not found")
                continue

            pf = pq.ParquetFile(src)
            seen = 0
            buffer: dict[str, list] = {name: [] for name in QUERIES_SCHEMA.names}

            for batch in pf.iter_batches(
                batch_size=2000,
                columns=["query_id", "query_type", "eng_query", "translated_query",
                         "eng_answer", "translated_answer", "is_selected"],
            ):
                for row in batch.to_pylist():
                    buffer["query_id"].append(row["query_id"])
                    buffer["lang"].append(lang)
                    buffer["query_type"].append(row["query_type"])
                    buffer["eng_query"].append(row["eng_query"])
                    buffer["translated_query"].append(row["translated_query"])
                    buffer["eng_answer"].append(row["eng_answer"])
                    buffer["translated_answer"].append(row["translated_answer"])
                    buffer["answerable"].append(any(row["is_selected"]))
                    seen += 1
                    if limit is not None and seen >= limit:
                        break

                writer.write_table(pa.Table.from_pydict(buffer, schema=QUERIES_SCHEMA),
                                   row_group_size=2000)
                for value in buffer.values():
                    value.clear()
                if limit is not None and seen >= limit:
                    break

            counts[lang] = seen
            print(f"  {lang}: {seen:,} queries")
    finally:
        writer.close()

    print(f"queries   : {sum(counts.values()):,} rows -> {dst} "
          f"({dst.stat().st_size / 1e6:,.0f} MB)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lang", default="hi",
                    help="shard to read the (identical) English side from")
    ap.add_argument("--langs", nargs="*", default=["hi", "bn", "ta", "mr"],
                    help="languages for the query/eval table")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--docs-only", action="store_true")
    args = ap.parse_args()

    build_docs(args.source_lang, args.limit)
    if not args.docs_only:
        build_queries(args.langs, args.limit)


if __name__ == "__main__":
    main()
