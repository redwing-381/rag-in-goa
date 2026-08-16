/**
 * Parse a fetch() body as server-sent events.
 *
 * Native EventSource is GET-only, so the live turn uses POST + this reader.
 */

export type SseEvent = {
  event: string;
  data: string;
};

export async function* readSSE(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const body = response.body;
  if (!body) throw new Error("The stream had no body.");

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const abort = () => {
    void reader.cancel();
  };
  signal?.addEventListener("abort", abort, { once: true });

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // sse-starlette uses CRLF; `\n\n` never matches `\r\n\r\n`.
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        const parsed = parseBlock(block);
        if (parsed) yield parsed;
      }
    }
    const tail = parseBlock(buffer);
    if (tail) yield tail;
  } finally {
    signal?.removeEventListener("abort", abort);
    try {
      reader.releaseLock();
    } catch {
      // cancel() already released the lock
    }
  }
}

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const raw of block.split(/\r?\n/)) {
    const line = raw.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}
