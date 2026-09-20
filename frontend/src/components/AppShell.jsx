import React, { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  ChevronLeft,
  ChevronRight,
  Bot,
  Files,
  FileText,
  LayoutDashboard,
  LogOut,
  Menu,
  Moon,
  Quote,
  Settings,
  Sun,
  Workflow,
  X,
} from "lucide-react";

import { Logo } from "./Logo";
import { cn } from "../utils/cn";
import { useAuthStore } from "../store/authStore";
import { initials } from "../utils/format";
import { applyTheme, getStoredTheme, toggleTheme } from "../utils/theme";
import { backendDataService } from "../services/backendData";

const STORAGE_LIMIT_BYTES = 500 * 1024 * 1024;

const items = [
  ["/dashboard", "Dashboard", LayoutDashboard, LayoutDashboard],
  ["/chat", "Chat & Ask", Bot, Bot],
  ["/documents", "Documents", Files, Files],
  ["/summaries", "Summaries", FileText, FileText],
  ["/comparisons", "Comparisons", Workflow, Workflow],
  ["/citations", "Citations", Quote, Quote],
  ["/settings", "Settings", Settings, Settings],
];

const meta = {
  "/dashboard": [
    "Dashboard Overview",
    "Comprehensive usage analytics, storage monitoring, and activity indicators.",
  ],
  "/chat": [
    "Chat & Ask AI",
    "Ask questions, explore insights, and run agentic reasoning over multiple files.",
  ],
  "/documents": [
    "Document Management",
    "Upload, manage, and inspect files. Documents are parsed into vector chunks instantly.",
  ],
  "/summaries": [
    "AI Document Summaries",
    "Generate document-level, executive, or comprehensive reports.",
  ],
  "/comparisons": [
    "AI Cross-Document Comparisons",
    "Contrast and correlate metrics, timelines, and insights across selected documents.",
  ],
  "/citations": [
    "Sources & Citations Library",
    "Review document chunks and verbatim snippets retrieved by RAG.",
  ],
  "/settings": ["System Settings", "Manage account security and RAG preferences."],
  "/profile": ["Profile", "Manage your DocMind AI account details."],
};

export function AppShell() {
  const loc = useLocation();
  const nav = useNavigate();
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);

  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem("docmind_sidebar_collapsed") === "1",
  );
  const [mobileOpen, setMobileOpen] = useState(false);
  const [theme, setTheme] = useState(() => getStoredTheme());
  const [storageUsedBytes, setStorageUsedBytes] = useState(0);

  useEffect(() => {
    localStorage.setItem("docmind_sidebar_collapsed", collapsed ? "1" : "0");
  }, [collapsed]);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    const syncTheme = (event) => {
      setTheme(event.detail === "light" ? "light" : "dark");
    };

    window.addEventListener("docmind:theme", syncTheme);
    return () => window.removeEventListener("docmind:theme", syncTheme);
  }, []);

  useEffect(() => {
    let active = true;

    void backendDataService
      .dashboard(7)
      .then((data) => {
        if (!active) return;
        setStorageUsedBytes(
          Math.max(0, Number(data?.metrics?.storage_used_bytes || 0)),
        );
      })
      .catch(() => {
        // Preserve the shell if dashboard analytics are temporarily unavailable.
      });

    return () => {
      active = false;
    };
  }, [loc.pathname, user?.id]);

  const [title, subtitle] = meta[loc.pathname] || ["DocMind AI", "Agentic RAG"];
  const width = collapsed ? 72 : 244;
  const storagePercent = Math.min(
    100,
    (storageUsedBytes / STORAGE_LIMIT_BYTES) * 100,
  );
  const storageLabel =
    storageUsedBytes > 0 && storagePercent < 1
      ? "<1%"
      : `${Math.round(storagePercent)}%`;
  const storageBarPercent =
    storageUsedBytes > 0 ? Math.max(storagePercent, 1) : 0;

  return (
    <div className="app-shell bg-ink-950 text-slate-100">
      {mobileOpen && (
        <button
          className="mobile-overlay lg:hidden"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        style={{ width }}
        className={cn(
          "desktop-sidebar sidebar-motion fixed inset-y-0 left-0 z-40 flex flex-col border-r border-slate-800 bg-[#03091a]",
          mobileOpen && "mobile-open",
        )}
      >
        <div
          className={cn(
            "relative flex h-[60px] items-center border-b border-slate-800",
            collapsed ? "justify-center px-2" : "justify-between px-3",
          )}
        >
          <Logo compact={collapsed} />

          <button
            className="rounded-md border border-slate-800 p-1.5 text-slate-400 hover:text-white lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>

          <button
            type="button"
            className={cn(
              "hidden h-8 w-8 shrink-0 place-items-center rounded-lg border border-slate-800 bg-ink-900 p-0 text-slate-400 shadow-sm transition hover:border-violet-700/60 hover:text-white lg:grid",
              collapsed && "absolute -right-4 top-1/2 z-50 -translate-y-1/2",
            )}
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="block h-4 w-4" />
            ) : (
              <ChevronLeft className="block h-4 w-4" />
            )}
          </button>
        </div>

        <nav className="space-y-1 px-2 py-4">
          {items.map(([to, label, DarkIcon, LightIcon]) => {
            const Icon = theme === "light" ? LightIcon : DarkIcon;

            return (
            <NavLink
              key={to}
              to={to}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium",
                  isActive
                    ? "border border-violet-700/40 bg-violet-950/55 text-violet-200"
                    : "text-slate-400 hover:bg-slate-900 hover:text-white",
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
            );
          })}
        </nav>

        <div className="mt-auto border-t border-slate-800 p-3">
          <div
            className="mb-3 rounded-lg border border-slate-800 bg-ink-900 p-2.5"
            title={`${storageUsedBytes.toLocaleString()} bytes of 500 MB used`}
          >
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>{collapsed ? "Storage" : "Storage Used"}</span>
              <span>{storageLabel}</span>
            </div>
            <div className="storage-usage-track mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div
                className="storage-usage-bar h-full rounded-full bg-slate-800 transition-[width] duration-300"
                style={{ width: `${storageBarPercent}%` }}
                aria-hidden="true"
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => nav("/profile")}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-violet-600 text-xs font-bold"
            >
              {initials(user?.username)}
            </button>

            {!collapsed && (
              <button
                onClick={() => nav("/profile")}
                className="min-w-0 flex-1 text-left"
              >
                <div className="truncate text-xs font-semibold">
                  {user?.username || "User"}
                </div>
                <div className="truncate text-[9px] text-slate-500">
                  {user?.email || ""}
                </div>
              </button>
            )}

            <button
              onClick={() => {
                logout();
                nav("/login");
              }}
              className="text-slate-500 hover:text-white"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      <div className="desktop-content content-motion" style={{ paddingLeft: width }}>
        <header className="sticky top-0 z-30 flex h-[60px] items-center border-b border-slate-800 bg-ink-950/95 px-3 sm:px-5">
          <button
            className="mr-3 rounded-lg border border-slate-800 p-2 lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-4 w-4" />
          </button>

          <div className="min-w-0">
            <h1 className="app-page-heading truncate text-base font-semibold">{title}</h1>
            <p className="hidden truncate text-[10px] text-slate-500 sm:block">
              {subtitle}
            </p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <div className="hidden rounded-full border border-slate-800 bg-ink-900 px-3 py-1 text-[10px] sm:block">
              <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400" />
              RAG Engine Active
            </div>

            <button
              type="button"
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              onClick={() => setTheme(toggleTheme())}
              className="grid h-8 w-8 place-items-center rounded-lg border border-slate-800"
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4 text-violet-300" />
              ) : (
                <Moon className="h-4 w-4 text-violet-600" />
              )}
            </button>

            <button
              onClick={() => nav("/profile")}
              className="header-account-avatar grid h-8 w-8 place-items-center rounded-full bg-slate-800 text-[10px] font-bold"
              aria-label="Open profile"
              title="Open profile"
            >
              {initials(user?.username)}
            </button>
          </div>
        </header>

        <main className="min-h-[calc(100vh-60px)]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
