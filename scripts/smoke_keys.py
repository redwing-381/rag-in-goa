"""Prove the two external providers actually work before we depend on them.

Both failures we care about are silent until runtime: a key that authenticates but
lacks credit, and a request shape the provider quietly reinterprets. So this hits
the real endpoints and asserts on the response body, not just the status code.

    python scripts/smoke_keys.py                 # both
    python scripts/smoke_keys.py --only llm
    python scripts/smoke_keys.py --only stt --audio /tmp/probe.wav

Generate a probe clip on macOS without extra tooling:

    say -o /tmp/probe.wav --data-format=LEI16@16000 "who wrote the book"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from ragoa.schemas import Chunk, ChunkingStrategy, Language, RetrievedChunk

# A self-contained fact the model cannot know from pretraining, so a correct
# answer proves it read our context rather than recalling something plausible.
PROBE_TEXT = (
    "The Konkan Vector Institute published its annual report in 2031. "
    "The report states that the institute indexed 4.2 million documents that year "
    "and that its founding director was Meera Shenoy."
)
PROBE_QUERY = "Who was the founding director of the Konkan Vector Institute?"
PROBE_ANSWER_KEY = "shenoy"
PROBE_SPEECH = "who wrote the book"


def probe_chunk() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="smoke-0",
        doc_id="smoke-doc",
        text=PROBE_TEXT,
        char_start=0,
        char_end=len(PROBE_TEXT),
        strategy=ChunkingStrategy.FIXED,
    )
    return RetrievedChunk(chunk=chunk, fused_score=1.0)


def check_llm(runs: int = 3) -> bool:
    from ragoa.harness.llm import OpenRouterLLM

    print("--- OpenRouter ---")
    llm = OpenRouterLLM()
    print(f"  model     : {llm.settings.llm_model}")
    print(f"  providers : {', '.join(llm.settings.llm_providers)}")

    retrieved = [probe_chunk()]
    ok = True

    # Repeat because the first call pays TLS setup and a cold upstream; a single
    # sample cannot tell that apart from a genuinely slow provider.
    for run in range(runs):
        payload, elapsed_ms = llm.answer(PROBE_QUERY, retrieved, Language.EN)
        label = "cold" if run == 0 else f"warm {run}"

        if payload.refusal_reason is not None:
            print(f"  {label:<6}: FAILED OVER to fallback ({payload.refusal_reason})")
            ok = False
            continue

        if PROBE_ANSWER_KEY not in payload.answer.lower():
            print(f"  {label:<6}: answer lacks '{PROBE_ANSWER_KEY}' -> {payload.answer[:80]!r}")
            ok = False
        if payload.citations != ["smoke-0"]:
            print(f"  {label:<6}: WARN unexpected citations {payload.citations}")

        stats = llm.upstream_stats()
        upstream = ""
        if stats:
            provider = stats.get("provider") or "?"
            parts = [provider]
            if stats.get("latency_ms") is not None:
                parts.append(f"ttft {stats['latency_ms']:,.0f}ms")
            if stats.get("generation_ms") is not None:
                parts.append(f"gen {stats['generation_ms']:,.0f}ms")
            if stats.get("tokens_completion"):
                parts.append(f"{stats['tokens_completion']} tok")
            if isinstance(stats.get("cost"), (int, float)):
                parts.append(f"${stats['cost']:.6f}")
            upstream = "   [" + ", ".join(parts) + "]"
            if provider.lower() not in [p.lower() for p in llm.settings.llm_providers]:
                upstream += " NOT PINNED"

        print(f"  {label:<6}: {elapsed_ms:>7,.0f} ms round trip{upstream}")

    print(f"  answer    : {payload.answer[:160]}")
    print(f"  {'PASS' if ok else 'FAIL'}: grounded generation over a synthetic context")
    return ok


def make_probe_wav() -> Path | None:
    """Synthesise a clip with macOS `say` so the check needs no fixture."""
    path = Path(tempfile.gettempdir()) / "ragoa_probe.wav"
    try:
        subprocess.run(
            ["say", "-o", str(path), "--data-format=LEI16@16000", PROBE_SPEECH],
            check=True, capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  could not synthesise audio with `say`: {exc}")
        return None
    return path if path.exists() else None


def check_stt(audio_path: Path | None) -> bool:
    from ragoa.stt.sarvam import SarvamSTT, STTError, wav_duration_seconds

    print("--- Sarvam ---")
    stt = SarvamSTT()
    print(f"  endpoint  : {stt.settings.sarvam_stt_url}")
    print(f"  model     : {stt.settings.sarvam_stt_model}  mode={stt.settings.sarvam_stt_mode}")

    path = audio_path or make_probe_wav()
    if path is None:
        print("  SKIP: no audio available; pass --audio path/to/clip.wav")
        return False

    audio = path.read_bytes()
    duration = wav_duration_seconds(audio)
    print(f"  clip      : {path.name}, {len(audio) / 1024:,.0f} KB"
          + (f", {duration:.1f}s" if duration else ", duration unknown"))

    try:
        result = stt.transcribe(audio, filename=path.name)
    except STTError as exc:
        print(f"  FAIL: {exc}")
        return False

    print(f"  latency   : {result.duration_ms:,.0f} ms")
    print(f"  transcript: {result.text!r}")
    print(f"  language  : {result.detected_language}  p={result.language_probability}")

    if not result.text:
        print("  FAIL: empty transcript")
        return False
    print("  PASS: audio in, text out")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["llm", "stt"])
    ap.add_argument("--audio", type=Path)
    args = ap.parse_args()

    results = {}
    if args.only in (None, "llm"):
        results["llm"] = check_llm()
        print()
    if args.only in (None, "stt"):
        results["stt"] = check_stt(args.audio)

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\n{', '.join(failed)} did not pass")
        return 1
    print(f"\nall clear: {', '.join(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
