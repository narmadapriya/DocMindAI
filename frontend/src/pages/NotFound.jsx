import React from "react";
import { Link } from "react-router-dom";
export default function NotFound() {
    return <div className="grid min-h-screen place-items-center bg-ink-950 text-center"><div><div className="text-6xl font-black text-violet-500">404</div><p className="mt-3 text-slate-400">Page not found.</p><Link to="/dashboard" className="mt-5 inline-block text-violet-300">Return to dashboard</Link></div></div>;
}
