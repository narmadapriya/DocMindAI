import React, { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { useAuthStore } from "./store/authStore";

const Login = lazy(() => import("./pages/Login"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword"));
const Register = lazy(() => import("./pages/Register"));
const Profile = lazy(() => import("./pages/Profile"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Documents = lazy(() => import("./pages/Documents"));
const DocumentViewer = lazy(() => import("./pages/DocumentViewer"));
const Chat = lazy(() => import("./pages/Chat"));
const Summaries = lazy(() => import("./pages/Summaries"));
const Comparisons = lazy(() => import("./pages/Comparisons"));
const Citations = lazy(() => import("./pages/Citations"));
const Settings = lazy(() => import("./pages/Settings"));
const Upload = lazy(() => import("./pages/Upload"));
const NotFound = lazy(() => import("./pages/NotFound"));

function RouteFallback() {
  return (
    <div className="grid min-h-[220px] place-items-center text-xs text-slate-500">
      Loading DocMind AI…
    </div>
  );
}

export default function App() {
  const initialize = useAuthStore((state) => state.initialize);
  const logout = useAuthStore((state) => state.logout);

  useEffect(() => {
    void initialize();

    const unauthorized = () => logout();
    window.addEventListener("docmind:unauthorized", unauthorized);

    return () =>
      window.removeEventListener("docmind:unauthorized", unauthorized);
  }, [initialize, logout]);

  return (
    <Suspense fallback={<RouteFallback />}>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />

        <Route element={<ProtectedRoute />}>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/documents/:id" element={<DocumentViewer />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/summaries" element={<Summaries />} />
            <Route path="/comparisons" element={<Comparisons />} />
            <Route path="/citations" element={<Citations />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/profile" element={<Profile />} />
          </Route>
        </Route>

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}
