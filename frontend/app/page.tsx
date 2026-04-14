"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://deep-forecast.onrender.com";

interface ForecastData {
  forecast_dates: string[];
  forecast_open: number[];
  forecast_close: number[];
  last_known_open: number;
  last_known_close: number;
}

interface ChartPoint {
  date: string;
  open: number | null;
  close: number | null;
  type: "forecast";
}

export default function Home() {
  const [data, setData]       = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [nDays, setNDays]     = useState(5);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Load cached forecast from static file for instant display
  const loadCachedForecast = async () => {
    try {
      const res = await fetch("/latest_forecast.json");
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  };

  // Fetch fresh forecast from API (background)
  const fetchFreshForecast = async () => {
    setIsRefreshing(true);
    try {
      const res = await fetch(`${API_URL}/forecast?n_days=${nDays}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: ForecastData = await res.json();
      setData(json);
      setError(null);
    } catch (e: any) {
      console.warn("Failed to refresh from API:", e.message);
      // Don't show error if we already have cached data
      if (!data) {
        setError(e.message || "Failed to fetch forecast");
      }
    } finally {
      setIsRefreshing(false);
    }
  };

  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    
    // Try cached forecast first (instant load)
    const cached = await loadCachedForecast();
    if (cached) {
      setData(cached);
      setLoading(false);
      // Refresh in background after loading cache
      fetchFreshForecast();
      return;
    }

    // Fallback: fetch from API if no cache
    try {
      const res = await fetch(`${API_URL}/forecast?n_days=${nDays}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: ForecastData = await res.json();
      setData(json);
    } catch (e: any) {
      setError(e.message || "Failed to fetch forecast");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchForecast(); }, []);

  // Build chart data: anchor point (last known) + forecast points
  const chartData: ChartPoint[] = data
    ? [
        {
          date: "Last Known",
          open: parseFloat(data.last_known_open.toFixed(2)),
          close: parseFloat(data.last_known_close.toFixed(2)),
          type: "forecast",
        },
        ...data.forecast_dates.map((d, i) => ({
          date: d,
          open: parseFloat(data.forecast_open[i].toFixed(2)),
          close: parseFloat(data.forecast_close[i].toFixed(2)),
          type: "forecast" as const,
        })),
      ]
    : [];

  // Metrics derived from forecast
  const openChange  = data ? ((data.forecast_open.at(-1)!  - data.last_known_open)  / data.last_known_open  * 100).toFixed(2) : null;
  const closeChange = data ? ((data.forecast_close.at(-1)! - data.last_known_close) / data.last_known_close * 100).toFixed(2) : null;

  return (
    <main className="min-h-screen bg-[#0f1117] text-white px-6 py-10 font-sans">

      {/* ── Header ─────────────────────────────────────────── */}
      <div className="max-w-5xl mx-auto mb-10 flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Stock Forecasting with GRU
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Residual Stacked GRU · Recursive multi-step prediction
          </p>
        </div>
        <Link
          href="/performance"
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg 
                     text-sm font-semibold transition-all whitespace-nowrap"
        >
          Model Metrics
        </Link>
      </div>

      {/* ── Controls ───────────────────────────────────────── */}
      <div className="max-w-5xl mx-auto flex items-center gap-4 mb-8">
        <label className="text-sm text-gray-400">Forecast days:</label>
        {[5].map((d) => (
          <button
            key={d}
            onClick={() => setNDays(d)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
              nDays === d
                ? "bg-indigo-600 text-white"
                : "bg-[#1e2130] text-gray-300 hover:bg-[#2a2f45]"
            }`}
          >
            {d}d
          </button>
        ))}
        <button
          onClick={fetchForecast}
          disabled={loading || isRefreshing}
          className="ml-auto px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                     rounded-lg text-sm font-semibold transition-all"
        >
          {loading ? "Loading…" : isRefreshing ? "Refreshing…" : "Run Forecast"}
        </button>
      </div>

      {/* ── Error ──────────────────────────────────────────── */}
      {error && (
        <div className="max-w-5xl mx-auto mb-6 bg-red-900/40 border border-red-500 
                        rounded-lg px-4 py-3 text-red-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* ── Metric Cards ───────────────────────────────────── */}
      {data && (
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Last Open",    value: `₹${data.last_known_open.toFixed(2)}`,  color: "text-white" },
            { label: "Last Close",   value: `₹${data.last_known_close.toFixed(2)}`, color: "text-white" },
            {
              label: `Open in ${nDays}d`,
              value: `₹${data.forecast_open.at(-1)!.toFixed(2)}`,
              sub: `${Number(openChange) >= 0 ? "+" : ""}${openChange}%`,
              color: Number(openChange) >= 0 ? "text-emerald-400" : "text-red-400",
            },
            {
              label: `Close in ${nDays}d`,
              value: `₹${data.forecast_close.at(-1)!.toFixed(2)}`,
              sub: `${Number(closeChange) >= 0 ? "+" : ""}${closeChange}%`,
              color: Number(closeChange) >= 0 ? "text-emerald-400" : "text-red-400",
            },
          ].map((m) => (
            <div key={m.label} className="bg-[#1e2130] rounded-xl px-5 py-4 border border-[#2a2f45]">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{m.label}</p>
              <p className={`text-xl font-bold ${m.color}`}>{m.value}</p>
              {m.sub && <p className={`text-sm mt-0.5 ${m.color}`}>{m.sub}</p>}
            </div>
          ))}
        </div>
      )}

      {/* ── Chart ──────────────────────────────────────────── */}
      <div className="max-w-5xl mx-auto bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6 mb-8">
        <h2 className="text-base font-semibold text-gray-200 mb-4">
          Open &amp; Close Price Forecast
        </h2>
        {loading && (
          <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
            Fetching forecast…
          </div>
        )}
        {!loading && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" />
              <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 12 }} />
              <YAxis
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                domain={["auto", "auto"]}
                tickFormatter={(v) => `₹${v}`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f1117", border: "1px solid #2a2f45", borderRadius: 8 }}
                labelStyle={{ color: "#e5e7eb" }}
                formatter={(value, name) => [
                  `₹${value ?? ""}`,
                  name ?? "",
                ]}
              />
              <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 13 }} />
              <ReferenceLine x="Last Known" stroke="#4b5563" strokeDasharray="4 4" label="" />
              <Line
                type="monotone" dataKey="open" name="Open"
                stroke="#6366f1" strokeWidth={2.5}
                dot={{ r: 4, fill: "#6366f1" }} activeDot={{ r: 6 }}
              />
              <Line
                type="monotone" dataKey="close" name="Close"
                stroke="#10b981" strokeWidth={2.5}
                dot={{ r: 4, fill: "#10b981" }} activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Forecast Table ─────────────────────────────────── */}
      {data && (
        <div className="max-w-5xl mx-auto bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6">
          <h2 className="text-base font-semibold text-gray-200 mb-4">Forecast Table</h2>
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="text-gray-500 uppercase text-xs border-b border-[#2a2f45]">
                <th className="pb-2">Date</th>
                <th className="pb-2 text-right">Open (₹)</th>
                <th className="pb-2 text-right">Close (₹)</th>
                <th className="pb-2 text-right">Open Δ%</th>
                <th className="pb-2 text-right">Close Δ%</th>
              </tr>
            </thead>
            <tbody>
              {data.forecast_dates.map((date, i) => {
                const prevOpen  = i === 0 ? data.last_known_open  : data.forecast_open[i - 1];
                const prevClose = i === 0 ? data.last_known_close : data.forecast_close[i - 1];
                const dOpen     = ((data.forecast_open[i]  - prevOpen)  / prevOpen  * 100).toFixed(2);
                const dClose    = ((data.forecast_close[i] - prevClose) / prevClose * 100).toFixed(2);
                return (
                  <tr key={date} className="border-b border-[#2a2f45] hover:bg-[#252b3b] transition-colors">
                    <td className="py-2.5 text-gray-300">{date}</td>
                    <td className="py-2.5 text-right text-white">{data.forecast_open[i].toFixed(2)}</td>
                    <td className="py-2.5 text-right text-white">{data.forecast_close[i].toFixed(2)}</td>
                    <td className={`py-2.5 text-right font-medium ${Number(dOpen)  >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {Number(dOpen)  >= 0 ? "+" : ""}{dOpen}%
                    </td>
                    <td className={`py-2.5 text-right font-medium ${Number(dClose) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {Number(dClose) >= 0 ? "+" : ""}{dClose}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Footer ─────────────────────────────────────────── */}
      <p className="max-w-5xl mx-auto mt-10 text-center text-xs text-gray-600">
        Powered by GRU · FastAPI · Render · Vercel &nbsp;·&nbsp; For educational purposes only. Not financial advice.
      </p>
    </main>
  );
}