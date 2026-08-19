const QUESTION = /\b(what|who|whom|whose|which|where|when|why|how|is|are|was|were|do|does|did|can|could|would|should)\b/;

const END_PHRASES = [
  "goodbye",
  "good bye",
  "bye",
  "bye bye",
  "end chat",
  "end the chat",
  "end conversation",
  "end the conversation",
  "stop",
  "stop listening",
  "stop the chat",
  "that's all",
  "thats all",
  "that is all",
  "that's it",
  "thats it",
  "i'm done",
  "i am done",
  "im done",
  "we're done",
  "we are done",
  "nothing else",
  "no more questions",
  "thank you goodbye",
  "thanks goodbye",
  "thank you bye",
  "thanks bye",
  "see you",
  "see you later",
  "quit",
  "exit",
  "cancel",
  "finish",
  "finished",
  "i want to stop",
  "please stop",
  "अलविदा",
  "बाय",
  "बात खत्म",
  "बात बंद करो",
  "धन्यवाद अलविदा",
  "বিদায়",
  "বাই",
  "வணக்கம் முடி",
  "முடி",
  "நன்றி வணக்கம்",
  "निरोप",
  "संवाद संपला",
];

const POLITE = /^(please|okay|ok|alright|yes|yeah|thanks|thank you|thankyou)\s+/;

function normalizeCommand(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function stripPolite(text: string): string {
  let current = text;
  for (let i = 0; i < 3; i += 1) {
    const next = current.replace(POLITE, "");
    if (next === current) break;
    current = next;
  }
  return current.trim();
}

/** True when the utterance is a close-the-chat command, not a question about those words. */
export function isEndCommand(raw: string): boolean {
  const text = normalizeCommand(raw);
  if (!text || text.split(" ").length > 8) return false;
  if (QUESTION.test(text)) return false;
  const core = stripPolite(text);
  return END_PHRASES.some(
    (phrase) => core === phrase || text === phrase || text.endsWith(` ${phrase}`),
  );
}
