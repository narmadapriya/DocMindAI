import React, { useEffect, useMemo, useState } from "react";
import { KeyRound, Save, Settings2, ShieldCheck, UserRound } from "lucide-react";
import { Button, Card } from "../components/ui";
import { PasswordVisibilityToggle } from "../components/PasswordVisibilityToggle";
import { metaService } from "../services/meta";
import { requestWithFallback, getErrorMessage } from "../services/api";
import { useAuthStore } from "../store/authStore";
import { applyTheme, getStoredTheme } from "../utils/theme";

const fallback = {
  model_name: "qwen2.5vl:3b",
  embedding_model: "all-MiniLM-L6-v2",
  temperature: 0,
  dark_mode: true,
  notifications_enabled: true,
  citations_enabled: true,
  top_k: 5,
};

const rules = (password) => ({
  length: password.length >= 8,
  upper: /[A-Z]/.test(password),
  lower: /[a-z]/.test(password),
  number: /\d/.test(password),
  special: /[^A-Za-z0-9]/.test(password),
});

const ruleLabels = [
  ["length", "Minimum 8 characters"],
  ["upper", "Uppercase letter (A-Z)"],
  ["lower", "Lowercase letter (a-z)"],
  ["number", "Number (0-9)"],
  ["special", "Special character"],
];

export default function Settings() {
  const user = useAuthStore((state) => state.user);
  const [tab, setTab] = useState("account");
  const [settings, setSettings] = useState(() => ({
    ...fallback,
    dark_mode: getStoredTheme() === "dark",
  }));
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState("info");
  const [passwords, setPasswords] = useState({
    current_password: "",
    new_password: "",
    confirm_password: "",
  });
  const [visible, setVisible] = useState({
    current: false,
    next: false,
    confirm: false,
  });

  useEffect(() => {
    void metaService
      .settings()
      .then((data) =>
        setSettings((current) => ({
          ...fallback,
          ...data,
          dark_mode: current.dark_mode,
        })),
      )
      .catch(() => {});
  }, []);

  const checks = useMemo(
    () => rules(passwords.new_password),
    [passwords.new_password],
  );
  const rulesPassed = Object.values(checks).filter(Boolean).length;
  const allRules = rulesPassed === ruleLabels.length;
  const passwordReady = Boolean(
    allRules &&
      passwords.new_password === passwords.confirm_password &&
      passwords.current_password,
  );

  const saveRag = async () => {
    setLoading(true);
    setMessage("");
    try {
      setSettings({
        ...settings,
        ...(await metaService.updateSettings(settings)),
      });
      setMessageType("success");
      setMessage("Settings saved.");
    } catch (error) {
      setMessageType("error");
      setMessage(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const clearPasswordForm = () => {
    setPasswords({
      current_password: "",
      new_password: "",
      confirm_password: "",
    });
    setVisible({
      current: false,
      next: false,
      confirm: false,
    });
    setMessage("");
    setMessageType("info");
  };

  const updatePassword = async () => {
    if (!passwordReady) return;

    setLoading(true);
    setMessage("");

    try {
      await requestWithFallback(["/api/users/password"], {
        method: "PUT",
        data: {
          current_password: passwords.current_password,
          new_password: passwords.new_password,
        },
      });
      setPasswords({
        current_password: "",
        new_password: "",
        confirm_password: "",
      });
      setMessageType("success");
      setMessage("Password updated successfully.");
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setMessageType("error");
      setMessage(errorMessage);

      if (errorMessage === "Current password is incorrect.") {
        window.alert("Current password is incorrect.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-pad p-5">
      <Card className="mb-4 flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <span className="settings-heading-icon-tile grid h-9 w-9 place-items-center rounded-lg bg-violet-600">
            <Settings2 className="settings-heading-icon h-5 w-5" />
          </span>
          <h2 className="text-base font-bold">Account & Password Settings</h2>
        </div>

        <div className="flex rounded-lg border border-slate-800 bg-ink-900 p-1 text-[10px]">
          <button
            onClick={() => setTab("account")}
            className={`rounded-md px-3 py-2 ${
              tab === "account"
                ? "settings-tab-active bg-violet-950 text-violet-300"
                : "text-slate-400"
            }`}
          >
            Account & Password
          </button>
          <button
            onClick={() => setTab("rag")}
            className={`rounded-md px-3 py-2 ${
              tab === "rag"
                ? "settings-tab-active bg-violet-950 text-violet-300"
                : "text-slate-400"
            }`}
          >
            AI & RAG Parameters
          </button>
        </div>
      </Card>

      {tab === "account" ? (
        <div className="settings-grid grid gap-4 xl:grid-cols-[.82fr_1.18fr]">
          <div className="space-y-4">
            <Card className="p-5">
              <div className="mb-5 flex items-center gap-2 text-xs font-semibold">
                <UserRound className="settings-profile-icon h-4 w-4 text-violet-300" />
                Account Profile
              </div>
              <div className="space-y-4 text-xs">
                <div>
                  <div className="label">Full Name</div>
                  <div className="font-semibold">{user?.username || "User"}</div>
                </div>
                <div>
                  <div className="label">Official Email Address</div>
                  <div className="rounded-lg border border-slate-800 bg-[#03091a] p-3">
                    {user?.email}
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <div className="mb-4 flex items-center gap-2 text-xs font-semibold">
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
                Password Security Policy
              </div>
              <p className="text-[11px] leading-5 text-slate-400">
                All passwords must satisfy the 8-4 security rule.
              </p>
              <div className="mt-4 rounded-lg border border-slate-800 bg-[#03091a] p-4 text-[11px] leading-6">
                <b>The Core Formula:</b>
                <br />• Minimum 8 characters
                <br />• Uppercase and lowercase letters
                <br />• Number
                <br />• Special character
              </div>
            </Card>
          </div>

          <Card className="p-5">
            <div className="mb-5 flex items-center gap-2 border-b border-slate-800 pb-4 text-xs font-semibold">
              <KeyRound className="h-4 w-4 text-violet-300" />
              Update Password
            </div>

            {[
              ["Current Password", "current_password", "current"],
              ["New Password", "new_password", "next"],
              ["Confirm New Password", "confirm_password", "confirm"],
            ].map(([label, key, visibleKey]) => (
              <div className="mb-4" key={key}>
                <label className="label">{label} *</label>
                <div className="relative">
                  <input
                    className="field pr-11"
                    type={visible[visibleKey] ? "text" : "password"}
                    autoComplete={
                      key === "current_password"
                        ? "current-password"
                        : "new-password"
                    }
                    value={passwords[key]}
                    onChange={(event) =>
                      setPasswords({
                        ...passwords,
                        [key]: event.target.value,
                      })
                    }
                    placeholder={label}
                  />
                  <PasswordVisibilityToggle
                    visible={visible[visibleKey]}
                    onToggle={() =>
                      setVisible({
                        ...visible,
                        [visibleKey]: !visible[visibleKey],
                      })
                    }
                  />
                </div>
              </div>
            ))}

            <div className="password-rules-card mb-4 rounded-xl border border-violet-800/60 bg-violet-950/15 p-4">
              <div className="mb-3 flex items-center justify-between text-[11px] font-semibold">
                <span>The Core Formula (The 8-4 Rule)</span>
                <span
                  className={`rounded px-2 py-1 text-[10px] ${
                    allRules
                      ? "password-rules-badge-complete bg-violet-950 text-violet-300"
                      : "bg-amber-950 text-amber-300"
                  }`}
                >
                  {rulesPassed}/5 RULES MET
                </span>
              </div>

              <div className="grid gap-2 sm:grid-cols-2">
                {ruleLabels.map(([key, label]) => (
                  <div
                    key={key}
                    className={`rounded-md border p-2 text-[11px] ${
                      checks[key]
                        ? "password-rule-success border-emerald-900 bg-emerald-950/15 text-emerald-400"
                        : "border-slate-800 text-slate-500"
                    }`}
                  >
                    {checks[key] ? "✓" : "×"} &nbsp;{label}
                  </div>
                ))}
              </div>
            </div>

            {message && (
              <p
                role={messageType === "error" ? "alert" : "status"}
                className={`settings-feedback mb-3 rounded-lg border p-2.5 text-[11px] ${
                  messageType === "error"
                    ? "auth-feedback-error border-rose-900/70 bg-rose-950/20 text-rose-300"
                    : messageType === "success"
                      ? "auth-feedback-success border-emerald-900/70 bg-emerald-950/20 text-emerald-300"
                      : "border-slate-800 text-slate-400"
                }`}
              >
                {message}
              </p>
            )}

            <div className="flex gap-2">
              <Button
                className="flex-1"
                loading={loading}
                disabled={!passwordReady}
                onClick={updatePassword}
              >
                <Save className="h-4 w-4" />
                Save New Password
              </Button>

              <Button
                type="button"
                variant="secondary"
                disabled={loading}
                onClick={clearPasswordForm}
                className="px-3 text-[11px]"
              >
                Clear
              </Button>
            </div>
          </Card>
        </div>
      ) : (
        <div className="settings-grid grid gap-4 xl:grid-cols-[1fr_380px]">
          <Card className="p-5">
            <h2 className="text-sm font-semibold">General RAG Parameters</h2>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div>
                <label className="label">LLM Model</label>
                <input
                  className="field"
                  value={settings.model_name || ""}
                  onChange={(event) =>
                    setSettings({ ...settings, model_name: event.target.value })
                  }
                />
              </div>
              <div>
                <label className="label">Embedding Model</label>
                <input
                  className="field"
                  value={settings.embedding_model || ""}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      embedding_model: event.target.value,
                    })
                  }
                />
              </div>
              <div>
                <label className="label">Temperature</label>
                <input
                  className="w-full accent-violet-500"
                  type="range"
                  min="0"
                  max="2"
                  step=".1"
                  value={settings.temperature ?? 0}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      temperature: Number(event.target.value),
                    })
                  }
                />
              </div>
              <div>
                <label className="label">Top K</label>
                <input
                  className="w-full accent-violet-500"
                  type="range"
                  min="1"
                  max="20"
                  value={settings.top_k ?? 5}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      top_k: Number(event.target.value),
                    })
                  }
                />
              </div>
            </div>
            <div className="mt-5 text-right">
              <Button loading={loading} onClick={saveRag}>
                <Save className="h-4 w-4" />
                Save Parameters
              </Button>
            </div>
          </Card>

          <Card className="h-fit p-5">
            <h2 className="text-sm font-semibold">System Toggles</h2>
            <div className="mt-5 space-y-4">
              {[
                ["Dark Mode", "dark_mode"],
                ["Notifications", "notifications_enabled"],
                ["Source Citations", "citations_enabled"],
              ].map(([label, key]) => (
                <label
                  key={key}
                  className="flex items-center justify-between text-xs"
                >
                  <span>{label}</span>
                  <input
                    type="checkbox"
                    checked={Boolean(settings[key])}
                    onChange={(event) => {
                      const next = {
                        ...settings,
                        [key]: event.target.checked,
                      };
                      setSettings(next);
                      if (key === "dark_mode") {
                        applyTheme(event.target.checked ? "dark" : "light");
                      }
                    }}
                    className="accent-violet-500"
                  />
                </label>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
