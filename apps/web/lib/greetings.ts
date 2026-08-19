import type { Language } from "./types";

/** Opening and closing lines, spoken with the same TTS path as answers.

Spoken locally so a greeting does not wait on an LLM round-trip. The model
still handles the questions in the middle of the conversation.
*/
export const OPENING: Record<Language, string> = {
  en: "Hello. I can look up answers from the documents for you. What would you like to know?",
  hi: "नमस्ते। मैं दस्तावेज़ों से आपके सवालों के जवाब ढूँढने में मदद करूँगा। आप क्या जानना चाहते हैं?",
  bn: "নমস্কার। আমি নথি থেকে আপনার প্রশ্নের উত্তর খুঁজে দিতে এখানে আছি। আপনি কী জানতে চান?",
  ta: "வணக்கம். ஆவணங்களிலிருந்து உங்கள் கேள்விகளுக்கு பதில் தேட நான் இங்கே இருக்கிறேன். நீங்கள் என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
  mr: "नमस्कार. दस्तऐवजांमधून तुमच्या प्रश्नांची उत्तरे शोधण्यासाठी मी इथे आहे. तुम्हाला काय जाणून घ्यायचे आहे?",
};

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

export const CLOSING: Record<Language, string> = {
  en: "Thanks for chatting. Start again whenever you have another question. Goodbye.",
  hi: "बात करने के लिए धन्यवाद। अगर और कुछ चाहिए हो तो नई बातचीत शुरू करें। अलविदा।",
  bn: "কথা বলার জন্য ধন্যবাদ। আর কিছু লাগলে নতুন করে শুরু করুন। বিদায়।",
  ta: "உரையாடியதற்கு நன்றி. மேலும் ஏதேனும் வேண்டுமானால் புதிய உரையாடலைத் தொடங்குங்கள். வணக்கம்.",
  mr: "बोलण्यासाठी धन्यवाद. आणखी काही हवे असल्यास नवीन संवाद सुरू करा. नमस्कार.",
};
