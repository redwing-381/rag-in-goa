"use client";

import type { SessionTurn } from "@/lib/session";

export function SidebarHistory({
  turns,
  selectedId,
  onSelect,
  onNewChat,
}: {
  turns: SessionTurn[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}) {
  return (
    <aside className="flex h-full w-[220px] shrink-0 flex-col border-r border-edge bg-surface/40">
      <div className="flex items-center justify-between gap-2 border-b border-edge px-3 py-3">
        <p className="text-[11px] font-semibold tracking-wider text-white/40 uppercase">
          Session
        </p>
        <button
          type="button"
          onClick={onNewChat}
          className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-white/60
            transition hover:border-white/25 hover:text-white"
        >
          New chat
        </button>
      </div>
      <ol className="min-h-0 flex-1 space-y-1 overflow-y-auto p-2">
        {turns.length === 0 && (
          <li className="px-2 py-6 text-center text-xs text-white/30">
            Turns you ask will land here.
          </li>
        )}
        {turns.map((turn) => {
          const selected = turn.id === selectedId;
          return (
            <li key={turn.id}>
              <button
                type="button"
                onClick={() => onSelect(turn.id)}
                className={`w-full rounded-xl px-2.5 py-2 text-left transition ${
                  selected ? "bg-accent/15" : "hover:bg-white/5"
                }`}
              >
                <p className="truncate text-[11px] text-white/40">You</p>
                <p className="truncate text-xs text-white/80">{turn.query}</p>
                <p className="mt-1.5 truncate text-[11px] text-white/40">Agent</p>
                <p
                  className={`truncate text-xs ${
                    turn.refused ? "text-warm" : "text-white/70"
                  }`}
                >
                  {turn.answer}
                </p>
              </button>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
