import React, { useMemo, useState } from "react";
import {
  ArrowLeft,
  Check,
  KeyRound,
  LockKeyhole,
  Mail,
  X,
} from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { PasswordVisibilityToggle } from "../components/PasswordVisibilityToggle";
import { Button, Card } from "../components/ui";
import { authService } from "../services/auth";
import { getErrorMessage } from "../services/api";

function passwordRules(password) {
  return {
    length: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    number: /\d/.test(password),
    special: /[^A-Za-z0-9]/.test(password),
  };
}

const RULES = [
  ["length", "Minimum 8 characters"],
  ["upper", "Uppercase letter (A-Z)"],
  ["lower", "Lowercase letter (a-z)"],
  ["number", "Number (0-9)"],
  ["special", "Special character"],
];

export default function ForgotPassword() {
  const [searchParams] = useSearchParams();
  const token = (searchParams.get("token") || "").trim();
  const resetMode = Boolean(token);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [visible, setVisible] = useState({ password: false, confirm: false });
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [resetComplete, setResetComplete] = useState(false);

  const checks = useMemo(() => passwordRules(password), [password]);
  const rulesPassed = Object.values(checks).filter(Boolean).length;
  const allRulesPassed = rulesPassed === RULES.length;
  const passwordsMatch =
    confirmPassword.length > 0 && password === confirmPassword;
  const resetReady = allRulesPassed && passwordsMatch;

  const requestReset = async (event) => {
    event.preventDefault();
    setLoading(true);
    setFeedback(null);

    try {
      const response = await authService.requestPasswordReset(email);
      setFeedback({
        type: "success",
        text:
          response?.message ||
          "If an account exists for that email, a password reset link has been sent.",
      });
    } catch (error) {
      setFeedback({ type: "error", text: getErrorMessage(error) });
    } finally {
      setLoading(false);
    }
  };

  const confirmReset = async (event) => {
    event.preventDefault();
    if (!resetReady || resetComplete) return;

    setLoading(true);
    setFeedback(null);

    try {
      const response = await authService.resetPassword({
        token,
        newPassword: password,
      });
      setResetComplete(true);
      setPassword("");
      setConfirmPassword("");
      setFeedback({
        type: "success",
        text: response?.message || "Password reset successfully.",
      });
    } catch (error) {
      setFeedback({ type: "error", text: getErrorMessage(error) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page grid min-h-screen place-items-center px-4 py-8">
      <div className="w-full max-w-[430px]">
        <div className="mb-5 flex flex-col items-center text-center">
          <Logo compact />
          <h1 className="mt-3 text-lg font-bold text-white">Account Recovery</h1>
          <p className="mt-1 text-[11px] text-slate-500">
            {resetMode
              ? "Choose a new password using the secure recovery link."
              : "Recover access to your DocMind AI account securely."}
          </p>
        </div>

        <Card className="auth-card p-5 sm:p-6">
          <div className="flex items-center gap-2">
            <KeyRound className="h-5 w-5 text-violet-300" />
            <h2 className="text-base font-bold text-white">
              {resetMode ? "Reset Password" : "Forgot Password"}
            </h2>
          </div>

          {!resetMode ? (
            <>
              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                Enter the email address associated with your account. If the
                account exists, a short-lived reset link will be emailed to you.
              </p>

              <form onSubmit={requestReset} className="mt-5 space-y-4">
                <div>
                  <label className="label">Email Address</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                    <input
                      className="field pl-9"
                      type="email"
                      required
                      autoComplete="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      placeholder="you@example.com"
                    />
                  </div>
                </div>

                <Button type="submit" className="w-full" loading={loading}>
                  Send Reset Link
                </Button>
              </form>
            </>
          ) : (
            <>
              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                Enter a new password that satisfies all five security rules.
              </p>

              {!resetComplete && (
                <form onSubmit={confirmReset} className="mt-5 space-y-4">
                  <div>
                    <label className="label">New Password</label>
                    <div className="relative">
                      <LockKeyhole className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                      <input
                        className="field pl-9 pr-11"
                        type={visible.password ? "text" : "password"}
                        autoComplete="new-password"
                        required
                        value={password}
                        onChange={(event) => setPassword(event.target.value)}
                        placeholder="Create strong password"
                      />
                      <PasswordVisibilityToggle
                        visible={visible.password}
                        onToggle={() =>
                          setVisible((current) => ({
                            ...current,
                            password: !current.password,
                          }))
                        }
                      />
                    </div>
                  </div>

                  <div className="password-rules-card rounded-xl border border-violet-800/60 bg-violet-950/15 p-3">
                    <div className="mb-2 flex items-center justify-between gap-3 text-[11px] font-semibold">
                      <span>The 8-4 Password Rule</span>
                      <span
                        className={`rounded-md px-2 py-1 text-[10px] ${
                          allRulesPassed
                            ? "password-rules-badge-complete bg-violet-950/50 text-violet-300"
                            : "bg-amber-950/50 text-amber-300"
                        }`}
                      >
                        {rulesPassed}/5 RULES MET
                      </span>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {RULES.map(([key, label]) => (
                        <div
                          key={key}
                          className={`flex items-center gap-2 rounded-md border px-2.5 py-2 text-[11px] ${
                            checks[key]
                              ? "password-rule-success border-emerald-900/70 bg-emerald-950/15 text-emerald-400"
                              : "border-slate-800 text-slate-500"
                          }`}
                        >
                          {checks[key] ? (
                            <Check className="h-3 w-3 shrink-0" />
                          ) : (
                            <X className="h-3 w-3 shrink-0" />
                          )}
                          <span>{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="label">Confirm New Password</label>
                    <div className="relative">
                      <LockKeyhole className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                      <input
                        className={`field pl-9 pr-11 ${
                          confirmPassword && !passwordsMatch ? "border-rose-700" : ""
                        }`}
                        type={visible.confirm ? "text" : "password"}
                        autoComplete="new-password"
                        required
                        value={confirmPassword}
                        onChange={(event) => setConfirmPassword(event.target.value)}
                        placeholder="Confirm new password"
                      />
                      <PasswordVisibilityToggle
                        visible={visible.confirm}
                        onToggle={() =>
                          setVisible((current) => ({
                            ...current,
                            confirm: !current.confirm,
                          }))
                        }
                      />
                    </div>
                    {confirmPassword && (
                      <p
                        className={`mt-1.5 text-[10px] ${
                          passwordsMatch ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {passwordsMatch
                          ? "Passwords match."
                          : "Passwords do not match."}
                      </p>
                    )}
                  </div>

                  <Button
                    type="submit"
                    className="w-full"
                    loading={loading}
                    disabled={!resetReady}
                  >
                    Reset Password
                  </Button>
                </form>
              )}
            </>
          )}

          {feedback && (
            <div
              role="status"
              className={`auth-feedback mt-4 rounded-xl border p-3 text-[11px] leading-5 ${
                feedback.type === "success"
                  ? "auth-feedback-success border-emerald-900/70 bg-emerald-950/30 text-emerald-300"
                  : "auth-feedback-error border-rose-900/70 bg-rose-950/30 text-rose-300"
              }`}
            >
              {feedback.text}
            </div>
          )}

          <Link
            to="/login"
            className="auth-back-link mt-5 inline-flex items-center gap-2 text-xs font-medium text-violet-400 hover:text-violet-300"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to Login
          </Link>
        </Card>
      </div>
    </div>
  );
}
