import React from "react";
import { BrainCircuit } from "lucide-react";
export function Logo({ compact = false }) {
    return (<div className="flex items-center gap-2.5">
      <div className="grid h-9 w-9 place-items-center rounded-xl border border-violet-600/50 bg-violet-950/50 text-violet-300 shadow-glow">
        <BrainCircuit className="h-5 w-5"/>
      </div>
      {!compact && (<div className="leading-tight">
          <div className="text-sm font-bold text-white">DocMind AI</div>
          <div className="text-[9px] font-medium tracking-[.16em] text-violet-400">AGENTIC RAG</div>
        </div>)}
    </div>);
}
