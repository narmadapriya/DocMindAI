import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, LockKeyhole, Mail } from "lucide-react";
import { Logo } from "../components/Logo";
import { Button, Card } from "../components/ui";
import { PasswordVisibilityToggle } from "../components/PasswordVisibilityToggle";
import { useAuthStore } from "../store/authStore";

export default function Login() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const loading = useAuthStore((s) => s.loading);
  const error = useAuthStore((s) => s.error);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    try {
      await login({ email: email.trim().toLowerCase(), password });
      navigate("/dashboard", { replace: true });
    } catch { /* store exposes error */ }
  };

  return (
    <div className="auth-page grid min-h-screen place-items-center px-4 py-8">
      <div className="w-full max-w-[390px]">
        <div className="mb-5 flex flex-col items-center text-center">
          <Logo compact />
          <h1 className="mt-3 text-lg font-bold text-white">DocMind AI Gateway</h1>
          <p className="mt-1 text-[11px] text-slate-500">Analyze and search your multi-format documents with agentic retrieval models.</p>
        </div>
        <Card className="auth-card p-5 sm:p-6">
          <h2 className="text-base font-bold text-white">Welcome Back</h2>
          <p className="mt-1 text-[11px] text-slate-500">Sign in to your DocMind account to access RAG analytics.</p>
          <form className="mt-5 space-y-4" onSubmit={submit}>
            <div><label className="label">Email Address</label><div className="relative"><Mail className="absolute left-3 top-3 h-4 w-4 text-slate-500"/><input className="field pl-9" type="email" name="email" autoComplete="username" required value={email} onChange={(e)=>setEmail(e.target.value)} placeholder="you@example.com"/></div></div>
            <div><div className="flex items-center justify-between"><label className="label">Password</label><Link to="/forgot-password" className="auth-recovery-link text-xs text-violet-400 hover:text-violet-300">Forgot?</Link></div><div className="relative"><LockKeyhole className="absolute left-3 top-3 h-4 w-4 text-slate-500"/><input className="field pl-9 pr-11" type={showPassword?"text":"password"} name="password" autoComplete="current-password" required value={password} onChange={(e)=>setPassword(e.target.value)} placeholder="••••••••"/><PasswordVisibilityToggle visible={showPassword} onToggle={()=>setShowPassword(v=>!v)} /></div></div>
            {error && <div className="rounded-lg border border-rose-900 bg-rose-950/20 p-2.5 text-xs text-rose-300">{error}</div>}
            <Button type="submit" loading={loading} className="w-full">Sign In <ArrowRight className="h-4 w-4"/></Button>
          </form>
          <p className="mt-6 text-center text-[11px] text-slate-400">Don't have an account? <Link className="font-semibold text-violet-400" to="/register">Sign Up Free</Link></p>
        </Card>
      </div>
    </div>
  );
}
