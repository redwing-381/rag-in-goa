import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "RAG in Goa — voice questions over MS MARCO-XI",
  description:
    "Speak a question in English, Hindi, Bengali, Tamil or Marathi. Retrieval stays under a 200ms budget, and the system refuses when the corpus cannot support an answer.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
