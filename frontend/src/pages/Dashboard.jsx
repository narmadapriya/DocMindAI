import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  Clock3,
  Database,
  FileText,
  HardDrive,
  Layers3,
  MessageSquareText,
  Network,
  Upload,
  Workflow,
} from "lucide-react";

import { useNavigate } from "react-router-dom";
import { Button, Card } from "../components/ui";
import { backendDataService } from "../services/backendData";
import { formatBytes, formatDate } from "../utils/format";
import { getErrorMessage } from "../services/api";

const CHART_COLORS = [
  "#8b5cf6",
  "#06b6d4",
  "#22c55e",
  "#f59e0b",
  "#ef4444",
  "#3b82f6",
];

function displayNumber(value) {
  return value == null ? "—" : Number(value).toLocaleString();
}

function displayResponse(ms) {
  return ms == null ? "—" : `${(Number(ms) / 1000).toFixed(1)}s`;
}

function MetricCard({ label, value, hint, Icon, tone = "violet" }) {
  const tones = {
    violet: "border-violet-900/60 bg-violet-950/35 text-violet-300",
    emerald: "border-emerald-900/60 bg-emerald-950/35 text-emerald-300",
    blue: "border-blue-900/60 bg-blue-950/35 text-blue-300",
    amber: "border-amber-900/60 bg-amber-950/35 text-amber-300",
  };

  return (
    <Card className="dashboard-metric p-4">
      <div className="flex items-start gap-3">
        <span
          className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg border ${tones[tone]}`}
        >
          <Icon className="h-5 w-5" />
        </span>

        <div className="min-w-0 flex-1">
          <div className="truncate text-[10px] text-slate-400">{label}</div>
          <div className="mt-2 text-2xl font-bold tracking-tight text-white">
            {value}
          </div>
          <div className="mt-1 text-[10px] text-slate-500">{hint}</div>
        </div>
      </div>
    </Card>
  );
}

function StatusBadge({ status }) {
  const ok = String(status || "").toLowerCase() === "operational";
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[9px] font-medium ${
        ok
          ? "border-emerald-900/70 bg-emerald-950/30 text-emerald-400"
          : "border-amber-900/70 bg-amber-950/30 text-amber-300"
      }`}
    >
      {ok ? "Operational" : "Unavailable"}
    </span>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    void backendDataService
      .dashboard(7)
      .then((data) => {
        if (!active) return;
        setPayload(data);
        setError("");
      })
      .catch((requestError) => {
        if (!active) return;
        setError(getErrorMessage(requestError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const metrics = payload?.metrics || {};
  const trend = payload?.queries_over_time || [];
  const documentTypes = payload?.document_types || [];
  const recentActivity = payload?.recent_activity || [];
  const systemStatus = payload?.system_status || [];

  const totalDocuments = Number(metrics.total_documents || 0);

  const typeRows = useMemo(() => {
    const preferredOrder = ["PDF", "DOCX", "CSV", "XLSX", "TXT"];

    return [...documentTypes]
      .map((item) => ({
        ...item,
        name: String(item.name || "OTHER").toUpperCase(),
        percent:
          totalDocuments > 0
            ? Math.round((Number(item.value || 0) / totalDocuments) * 100)
            : 0,
      }))
      .sort((left, right) => {
        const leftIndex = preferredOrder.indexOf(left.name);
        const rightIndex = preferredOrder.indexOf(right.name);
        const safeLeft = leftIndex === -1 ? 999 : leftIndex;
        const safeRight = rightIndex === -1 ? 999 : rightIndex;
        return safeLeft - safeRight || left.name.localeCompare(right.name);
      });
  }, [documentTypes, totalDocuments]);

  const cards = [
    {
      label: "Total Documents",
      value: displayNumber(metrics.total_documents),
      hint: "Indexed repository documents",
      Icon: FileText,
      tone: "violet",
    },
    {
      label: "Chunks Indexed",
      value: displayNumber(metrics.chunks_indexed),
      hint: "Persisted PostgreSQL chunks",
      Icon: Layers3,
      tone: "violet",
    },
    {
      label: "Queries Asked",
      value: displayNumber(metrics.queries_asked),
      hint: "Persisted RAG query events",
      Icon: MessageSquareText,
      tone: "violet",
    },
    {
      label: "Storage Used",
      value:
        metrics.storage_used_bytes == null
          ? "—"
          : formatBytes(metrics.storage_used_bytes),
      hint: "Sum of persisted document file sizes",
      Icon: HardDrive,
      tone: "emerald",
    },
    {
      label: "Embeddings Generated",
      value: displayNumber(metrics.embeddings_generated),
      hint: "Vectors in authenticated Chroma collection",
      Icon: Network,
      tone: "violet",
    },
    {
      label: "Avg. Response Time",
      value: displayResponse(metrics.avg_response_time_ms),
      hint: "Last 7 days from analytics",
      Icon: Clock3,
      tone: "amber",
    },
  ];

  return (
    <div className="dashboard-page space-y-4 p-3 sm:p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Dashboard Overview</h2>
          <p className="mt-1 text-[11px] text-slate-500">
            Comprehensive usage analytics, system monitoring, and activity overview.
          </p>
        </div>

        <Button onClick={() => navigate("/documents")} className="self-start sm:self-auto">
          <Upload className="h-4 w-4" />
          Upload Document
        </Button>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-900/70 bg-rose-950/20 p-3 text-xs text-rose-400">
          Dashboard analytics endpoint error: {error}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        {cards.map((card) => (
          <MetricCard key={card.label} {...card} />
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.25fr_.8fr]">
        <Card className="p-4 sm:p-5">
          <div>
            <div className="text-sm font-semibold text-slate-200">Queries Over Time</div>
            <div className="mt-1 text-[10px] text-slate-500">
              Real RAG query events aggregated by day for the last 7 days.
            </div>
          </div>

          <div className="mt-4 h-[250px] sm:h-[300px]">
            {loading ? (
              <div className="grid h-full place-items-center text-xs text-slate-500">
                Loading query analytics…
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={trend}
                  margin={{ top: 12, right: 12, left: -8, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="queryAreaV7" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.42} />
                      <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.03} />
                    </linearGradient>
                  </defs>

                  <CartesianGrid vertical={false} stroke="var(--dm-chart-grid)" strokeOpacity={0.12} />
                  <XAxis
                    dataKey="day"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 10, fill: "var(--dm-muted)" }}
                  />
                  <YAxis
                    allowDecimals={false}
                    domain={[0, "auto"]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 10, fill: "var(--dm-muted)" }}
                  />
                  <Tooltip
                    formatter={(value) => [value, "Queries"]}
                    labelFormatter={(label, points) =>
                      points?.[0]?.payload?.date
                        ? `${label} • ${points[0].payload.date}`
                        : label
                    }
                    contentStyle={{
                      background: "var(--dm-tooltip-bg)",
                      border: "1px solid var(--dm-border)",
                      borderRadius: 10,
                      color: "var(--dm-text)",
                      fontSize: 11,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="queries"
                    stroke="#8b5cf6"
                    strokeWidth={3}
                    fill="url(#queryAreaV7)"
                    dot={false}
                    activeDot={{ r: 5, fill: "#8b5cf6", stroke: "#ffffff", strokeWidth: 1.5 }}
                    animationDuration={450}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        <Card className="p-4 sm:p-5">
          <div className="text-sm font-semibold text-slate-200">Documents by Type</div>
          <div className="mt-1 text-[10px] text-slate-500">
            Distribution from the authenticated PostgreSQL document repository.
          </div>

          <div className="documents-type-layout mt-4 grid min-h-[276px] items-center gap-3 sm:grid-cols-[minmax(210px,1fr)_minmax(150px,190px)]">
            <div className="relative mx-auto h-[220px] w-full max-w-[270px] sm:h-[236px]">
              {documentTypes.length ? (
                <>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={typeRows}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius="48%"
                        outerRadius="72%"
                        paddingAngle={0}
                        stroke="var(--dm-panel)"
                        strokeWidth={1}
                        isAnimationActive
                        animationDuration={450}
                      >
                        {typeRows.map((_, index) => (
                          <Cell
                            key={index}
                            fill={CHART_COLORS[index % CHART_COLORS.length]}
                          />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="pointer-events-none absolute inset-0 grid place-items-center">
                    <div className="text-center">
                      <div className="text-xl font-bold text-white">{totalDocuments}</div>
                      <div className="mt-0.5 text-[8px] uppercase tracking-wider text-slate-500">Files indexed</div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="grid h-full place-items-center text-xs text-slate-600">
                  No documents found.
                </div>
              )}
            </div>

            <div className="space-y-3 sm:pr-1">
              {typeRows.map((item, index) => (
                <div key={item.name} className="flex items-center gap-2.5 text-[11px]">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: CHART_COLORS[index % CHART_COLORS.length] }}
                  />
                  <span className="min-w-0 flex-1 truncate font-medium text-slate-300">
                    {item.name} ({item.value})
                  </span>
                  <span className="tabular-nums text-slate-400">{item.percent}%</span>
                </div>
              ))}

              <div className="border-t border-slate-800 pt-3 text-[10px] text-slate-500">
                Total: {totalDocuments} documents
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr_.9fr]">
        <Card className="p-4 sm:p-5">
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm font-semibold text-slate-200">Recent Activity</div>
            <button
              type="button"
              onClick={() => navigate("/documents")}
              className="text-[10px] font-medium text-violet-400 hover:text-violet-300"
            >
              View all documents →
            </button>
          </div>

          <div className="mt-3 divide-y divide-slate-800">
            {recentActivity.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => navigate(`/documents/${item.id}`)}
                data-document-type={String((item.filename || "document").split(".").pop() || "document").toUpperCase()}
                className="document-card-ui flex w-full items-center gap-3 py-2.5 text-left hover:bg-slate-900/40"
              >
                <span className="document-card-icon-wrap grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-slate-800 bg-slate-900">
                  <FileText className="document-card-icon h-4 w-4 text-violet-300" />
                </span>
                <span className="document-card-copy min-w-0 flex-1">
                  <span className="document-card-title block truncate text-[11px] font-medium text-slate-200">
                    {item.filename} uploaded
                  </span>
                  <span className="document-card-meta mt-0.5 block text-[9px] text-slate-600">
                    {formatDate(item.created_at)} • {item.chunk_count} chunks
                  </span>
                </span>
                <StatusBadge status={item.status === "ready" ? "operational" : "unavailable"} />
              </button>
            ))}

            {!recentActivity.length && (
              <div className="py-8 text-center text-xs text-slate-600">
                No document activity is stored yet.
              </div>
            )}
          </div>
        </Card>

        <Card className="p-4 sm:p-5">
          <div className="text-sm font-semibold text-slate-200">System Status</div>
          <div className="mt-4 space-y-2">
            {systemStatus.map((item) => {
              const Icon =
                item.name === "PostgreSQL"
                  ? Database
                  : item.name.includes("Vector") || item.name.includes("Embedding")
                    ? Network
                    : Workflow;

              return (
                <div
                  key={item.name}
                  className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-ink-800 px-3 py-2.5"
                >
                  <span className="flex min-w-0 items-center gap-2 text-[11px] text-slate-300">
                    <Icon className="h-4 w-4 shrink-0 text-slate-400" />
                    {item.name}
                  </span>
                  <StatusBadge status={item.status} />
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 px-1 py-2 text-[9px] text-slate-600">
        <span>DocMind AI Agentic RAG System • Multi-modal • Multi-document • Intelligent Retrieval</span>
        <span>DocMind AI</span>
      </div>
    </div>
  );
}
