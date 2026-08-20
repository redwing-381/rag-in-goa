"use client";

import type { SessionTurn } from "@/lib/session";

export function SidebarHistory({
  turns,
  selectedId,
  onSelect,
  onNewChat,
  className = "flex h-full w-[220px] shrink-0 flex-col border-r border-rule",
}: {
  turns: SessionTurn[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  className?: string;
}) {
  return (
    <aside className={className}>
      <div className="flex items-center justify-between gap-2 border-b border-rule px-4 py-4">
        <p className="font-mono text-[11px] tracking-wide text-muted">Contents</p>
        <button
          type="button"
          onClick={onNewChat}
          className="text-[11px] text-muted underline-offset-2 hover:text-ink hover:underline"
        >
          Clear session
        </button>
      </div>
      <ol className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {turns.length === 0 && (
          <li className="py-8 text-xs leading-relaxed text-muted">
            Questions you ask will be listed here.
          </li>
        )}
        {turns.map((turn, index) => {
          const selected = turn.id === selectedId;
          return (
            <li key={turn.id} className="border-b border-rule/70 py-2.5 last:border-b-0">
              <button
                type="button"
                onClick={() => onSelect(turn.id)}
                className={`w-full text-left ${selected ? "text-ink" : "text-muted hover:text-ink"}`}
              >
                <span className="font-mono text-[10px] tabular-nums text-muted">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <p className="mt-0.5 line-clamp-2 text-[13px] leading-snug">{turn.query}</p>
              </button>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
