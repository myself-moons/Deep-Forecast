"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://deep-forecast.onrender.com";

type MetricKey = "r2" | "mse" | "rmse" | "mae" | "dir_acc" | "dir_acc_large_moves";

interface MetricRow {
  metric: string;
  key: MetricKey;
  decimals: number;
  hasPriceColumns: boolean;
}

interface MetricsGroup {
  [key: string]: number | undefined;
}

interface MetricsResponse {
  open_ret: MetricsGroup;
  close_ret: MetricsGroup;
  open_px: MetricsGroup;
  close_px: MetricsGroup;
}

interface ForecastApiResponse {
  metrics: MetricsResponse;
}

const metricRows: MetricRow[] = [
  { metric: "R²", key: "r2", decimals: 4, hasPriceColumns: true },
  { metric: "MSE", key: "mse", decimals: 4, hasPriceColumns: true },
  { metric: "RMSE", key: "rmse", decimals: 4, hasPriceColumns: true },
  { metric: "MAE", key: "mae", decimals: 4, hasPriceColumns: true },
  { metric: "Direction Accuracy", key: "dir_acc", decimals: 3, hasPriceColumns: false },
  { metric: "Direction Accuracy (Large)", key: "dir_acc_large_moves", decimals: 3, hasPriceColumns: false },
];

const fallbackMetrics = [
  { metric: "R²", open_ret: "-0.5826", close_ret: "-3.7990", open_px: "0.9231", close_px: "0.7927" },
  { metric: "MSE", open_ret: "0.0001", close_ret: "0.0004", open_px: "41621.4514", close_px: "110373.6803" },
  { metric: "RMSE", open_ret: "0.0120", close_ret: "0.0194", open_px: "204.0134", close_px: "332.2253" },
  { metric: "MAE", open_ret: "0.0097", close_ret: "0.0162", open_px: "164.0673", close_px: "276.3165" },
  { metric: "Direction Accuracy", open_ret: "0.568", close_ret: "0.517", open_px: "—", close_px: "—" },
  { metric: "Direction Accuracy (Large)", open_ret: "0.559", close_ret: "0.407", open_px: "—", close_px: "—" },
];

function formatNumber(value: number | undefined, decimals: number) {
  if (value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

export default function Performance() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const response = await fetch(`${API_URL}/forecast?n_days=5`);
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        const data = (await response.json()) as ForecastApiResponse;
        setMetrics(data.metrics);
      } catch (err: any) {
        setError(err?.message || "Unable to load metrics");
      }
    }

    fetchMetrics();
  }, []);

  const tableMetrics = metrics
    ? metricRows.map((row) => ({
        metric: row.metric,
        open_ret: formatNumber(metrics.open_ret[row.key], row.decimals),
        close_ret: formatNumber(metrics.close_ret[row.key], row.decimals),
        open_px: row.hasPriceColumns
          ? formatNumber(metrics.open_px[row.key], 4)
          : "—",
        close_px: row.hasPriceColumns
          ? formatNumber(metrics.close_px[row.key], 4)
          : "—",
      }))
    : fallbackMetrics;

  return (
    <main className="min-h-screen bg-[#0f1117] text-white px-6 py-10 font-sans">
      {/* ── Navigation Bar ─────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto flex items-center justify-between mb-10">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Model Performance
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Residual Stacked GRU · Evaluation metrics &amp; visualizations
          </p>
        </div>
        <Link
          href="/"
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg 
                     text-sm font-semibold transition-all"
        >
          Back to Forecast
        </Link>
      </div>

      {/* ── Performance Chart ──────────────────────────────────── */}
      <div className="max-w-6xl mx-auto bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6 mb-8">
        <h2 className="text-base font-semibold text-gray-200 mb-4">
          Model Performance Analysis
        </h2>
        <div className="flex items-center justify-center bg-[#0f1117] rounded-lg p-4">
          <img
            src="/performance.png"
            alt="GRU Model Performance Visualization"
            className="max-w-full h-auto"
          />
        </div>
      </div>

      {/* ── Metrics Table ──────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-200">Detailed Metrics</h2>
          {error ? (
            <span className="text-xs text-orange-300">{error}</span>
          ) : (
            <span className="text-xs text-gray-400">
              {metrics ? "Live metrics from API" : "Using cached fallback metrics"}
            </span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="text-gray-500 uppercase text-xs border-b border-[#2a2f45]">
                <th className="pb-3 px-2">Metric</th>
                <th className="pb-3 px-2 text-right">Open Returns</th>
                <th className="pb-3 px-2 text-right">Close Returns</th>
                <th className="pb-3 px-2 text-right">Open Price</th>
                <th className="pb-3 px-2 text-right">Close Price</th>
              </tr>
            </thead>
            <tbody>
              {tableMetrics.map((row, idx) => (
                <tr
                  key={idx}
                  className="border-b border-[#2a2f45] hover:bg-[#252b3b] transition-colors"
                >
                  <td className="py-3 px-2 text-gray-300 font-medium">{row.metric}</td>
                  <td className="py-3 px-2 text-right text-white">{row.open_ret}</td>
                  <td className="py-3 px-2 text-right text-white">{row.close_ret}</td>
                  <td className="py-3 px-2 text-right text-white">{row.open_px}</td>
                  <td className="py-3 px-2 text-right text-white">{row.close_px}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Metrics Legend ─────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto mt-8 bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6">
        <h2 className="text-base font-semibold text-gray-200 mb-4">Metric Explanations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p className="font-semibold text-gray-100 mb-1">R² (Coefficient of Determination)</p>
            <p className="text-gray-400">Proportion of variance explained. Higher is better (max 1.0).</p>
          </div>
          <div>
            <p className="font-semibold text-gray-100 mb-1">MSE (Mean Squared Error)</p>
            <p className="text-gray-400">Average squared error. Lower is better.</p>
          </div>
          <div>
            <p className="font-semibold text-gray-100 mb-1">RMSE (Root Mean Squared Error)</p>
            <p className="text-gray-400">Square root of MSE. Same units as output. Lower is better.</p>
          </div>
          <div>
            <p className="font-semibold text-gray-100 mb-1">MAE (Mean Absolute Error)</p>
            <p className="text-gray-400">Average absolute error. Lower is better.</p>
          </div>
        </div>
      </div>

      {/* ── Model Info ─────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto mt-8 mb-10 bg-[#1e2130] rounded-2xl border border-[#2a2f45] p-6">
        <h2 className="text-base font-semibold text-gray-200 mb-3">Model Architecture</h2>
        <div className="text-sm text-gray-300 space-y-2">
          <p><span className="text-gray-400">Type:</span> <span className="text-white font-medium">Residual Stacked GRU</span></p>
          <p><span className="text-gray-400">Input:</span> <span className="text-white font-medium">40-day windows of log-returns</span></p>
          <p><span className="text-gray-400">Output:</span> <span className="text-white font-medium">Open & Close price log-returns</span></p>
          <p><span className="text-gray-400">Loss Function:</span> <span className="text-white font-medium">Huber Directional Loss (δ=0.05, direction_weight=0.25)</span></p>
          <p><span className="text-gray-400">Data:</span> <span className="text-white font-medium">84 months of historical data with 85% train / 15% test split</span></p>
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────── */}
      <p className="max-w-6xl mx-auto text-center text-xs text-gray-600">
        Powered by GRU · FastAPI · Render · Vercel &nbsp;·&nbsp; For educational purposes only. Not financial advice.
      </p>
    </main>
  );
}
