"""Throwaway probe: can we read the first N rows of a MSMARCO-XI validation
shard over HTTP range requests, or must we download the whole 0.46 GB file?

Measures bytes actually transferred, so the answer is evidence not guesswork.
"""

import io
import time

import pyarrow.parquet as pq
import requests

URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/"
    "resolve/main/validation/hinval.parquet"
)

WANT_COLUMNS = [
    "query",
    "Eng_Query",
    "Answer",
    "Eng_Answer",
    "query_id",
    "query_type",
    "passages",
]


class RangeFile(io.RawIOBase):
    """Minimal seekable file-like object backed by HTTP range requests."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.accepts_ranges = head.headers.get("Accept-Ranges", "none")
        self.pos = 0
        self.bytes_read = 0
        self.n_requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        resp.raise_for_status()
        data = resp.content
        self.pos += len(data)
        self.bytes_read += len(data)
        self.n_requests += 1
        return data


def main():
    f = RangeFile(URL)
    print(f"file size            : {f.size / 1e9:.3f} GB")
    print(f"Accept-Ranges        : {f.accepts_ranges}")

    t0 = time.perf_counter()
    pf = pq.ParquetFile(f)
    meta = pf.metadata
    footer_bytes = f.bytes_read
    print(f"footer read          : {footer_bytes / 1e6:.2f} MB "
          f"in {f.n_requests} requests, {time.perf_counter() - t0:.1f}s")
    print(f"rows                 : {meta.num_rows:,}")
    print(f"row groups           : {meta.num_row_groups}")
    print(f"columns              : {meta.num_columns}")

    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        compressed = sum(rg.column(c).total_compressed_size
                         for c in range(rg.num_columns))
        print(f"  row group {i}: {rg.num_rows:,} rows, "
              f"{rg.total_byte_size / 1e9:.2f} GB uncompressed, "
              f"{compressed / 1e9:.2f} GB compressed")

    print("\nschema top-level fields:")
    for name in meta.schema.names[:40]:
        print("   ", name)

    print("\n--- attempting first-batch read with column projection ---")
    before = f.bytes_read
    t1 = time.perf_counter()
    batches = pf.iter_batches(batch_size=64, columns=WANT_COLUMNS)
    first = next(batches)
    elapsed = time.perf_counter() - t1
    transferred = f.bytes_read - before
    print(f"got {first.num_rows} rows in {elapsed:.1f}s")
    print(f"bytes transferred    : {transferred / 1e6:.2f} MB "
          f"({100 * transferred / f.size:.1f}% of file)")
    print(f"total bytes           : {f.bytes_read / 1e6:.2f} MB in {f.n_requests} requests")

    row = first.to_pylist()[0]
    passages = row["passages"]
    print(f"\nsample row: query_id={row['query_id']} type={row['query_type']}")
    print(f"  query      : {row['query'][:90]}")
    print(f"  Eng_Query  : {row['Eng_Query'][:90]}")
    print(f"  n_passages : {len(passages['English_passages'])}")
    print(f"  is_selected: {passages['is_selected']}")
    eng = passages["English_passages"]
    tr = passages["Translated_passages"]
    print(f"  eng chars  : {[len(p) for p in eng]}")
    print(f"  doc chars  : eng={sum(len(p) for p in eng)} translated={sum(len(p) for p in tr)}")
    print(f"  eng[0]     : {eng[0][:160]}")
    print(f"  trans[0]   : {tr[0][:160]}")


if __name__ == "__main__":
    main()
