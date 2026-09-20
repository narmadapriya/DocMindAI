import React from "react";
import { LoaderCircle } from "lucide-react";
import { cn } from "../utils/cn";
export function Card({ className, ...props }) {
    return <div className={cn("panel", className)} {...props}/>;
}
export function Button({ children, className, loading, variant = "primary", ...props }) {
    const styles = {
        primary: "btn-primary",
        secondary: "btn-secondary",
        ghost: "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-800/70",
        danger: "inline-flex items-center justify-center gap-2 rounded-lg border border-rose-900/70 bg-rose-950/20 px-3 py-2 text-sm text-rose-300 hover:bg-rose-950/40",
    };
    return (<button className={cn(styles[variant], className)} disabled={loading || props.disabled} {...props}>
      {loading && <LoaderCircle className="h-4 w-4 animate-spin"/>}
      {children}
    </button>);
}
export function EmptyState({ icon, title, description, action, }) {
    return (<div className="flex min-h-[260px] flex-col items-center justify-center px-6 text-center">
      <div className="mb-3 text-slate-700">{icon}</div>
      <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
      <p className="mt-1 max-w-lg text-xs leading-5 text-slate-500">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>);
}
export function StatusPill({ status }) {
    const value = (status || "unknown").toLowerCase();
    const ready = ["ready", "processed", "success", "indexed"].includes(value);
    const failed = value === "failed";
    return (<span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold capitalize", ready && "border-emerald-900/80 bg-emerald-950/30 text-emerald-400", failed && "border-rose-900/80 bg-rose-950/30 text-rose-400", !ready && !failed && "border-amber-900/80 bg-amber-950/30 text-amber-300")}>
      <span className="h-1.5 w-1.5 rounded-full bg-current"/>
      {status || "unknown"}
    </span>);
}
