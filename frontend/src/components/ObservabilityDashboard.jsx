import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  BarChart2,
  CheckCircle2,
  Loader2,
  RefreshCw,
  TrendingUp,
  Wifi,
  WifiOff,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE = 'http://localhost:8000';
const POLL_INTERVAL_MS = 3000;

/* ─── Prometheus text-format parsers ─── */
function parsePrometheusText(text) {
  const result = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const spaceIdx = trimmed.lastIndexOf(' ');
    if (spaceIdx === -1) continue;
    const key = trimmed.substring(0, spaceIdx).trim().replace(/\{[^}]*\}/, '');
    const val = parseFloat(trimmed.substring(spaceIdx + 1));
    if (!isNaN(val)) result[key] = (result[key] ?? 0) + val;
  }
  return result;
}

function parseCategoryDistribution(text) {
  const dist = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    if (!trimmed.includes('model_predictions_total')) continue;
    const labelMatch = trimmed.match(/\{([^}]*)\}/);
    if (!labelMatch) continue;
    const categoryMatch = labelMatch[1].match(/category="([^"]+)"/);
    const spaceIdx = trimmed.lastIndexOf(' ');
    const val = parseFloat(trimmed.substring(spaceIdx + 1));
    if (categoryMatch && !isNaN(val)) {
      const cat = categoryMatch[1];
      dist[cat] = (dist[cat] ?? 0) + val;
    }
  }
  return dist;
}

const CATEGORY_COLORS = {
  Database:       '#6366f1',
  Network:        '#8b5cf6',
  Infrastructure: '#f59e0b',
  Application:    '#10b981',
  Security:       '#ef4444',
};
const DEFAULT_COLOR = '#94a3b8';

/* ─── Stat card ─── */
function StatCard({ icon: Icon, iconBg, iconColor, label, value, sub }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-xl shrink-0 ${iconBg}`}>
        <Icon className={`w-5 h-5 ${iconColor}`} />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-slate-400 font-medium uppercase tracking-wide truncate">{label}</p>
        <p className="text-2xl font-bold text-slate-800 mt-0.5 leading-none">{value}</p>
        {sub && <p className="text-xs text-slate-400 mt-0.5 font-mono truncate">{sub}</p>}
      </div>
    </div>
  );
}

export default function ObservabilityDashboard() {
  const [metrics, setMetrics]         = useState(null);
  const [history, setHistory]         = useState([]);       // [{time, totalPredictions, avgConfidence}]
  const [categoryDist, setCategoryDist] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const intervalRef                   = useRef(null);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const text = await res.text();
      const parsed = parsePrometheusText(text);

      const totalPredictions = parsed['model_predictions_total'] ?? 0;
      const avgConfidence    = parsed['model_prediction_confidence_score'] ?? 0;

      setMetrics({ totalPredictions, avgConfidence });

      const timeLabel = new Date().toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
      setHistory((prev) => [
        ...prev.slice(-19),
        { time: timeLabel, totalPredictions, avgConfidence: parseFloat((avgConfidence * 100).toFixed(1)) },
      ]);

      const dist = parseCategoryDistribution(text);
      setCategoryDist(Object.entries(dist).map(([name, value]) => ({ name, value })));
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Cannot connect to the backend at http://localhost:8000. Make sure the FastAPI server is running.');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchMetrics(); }, [fetchMetrics]);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchMetrics, POLL_INTERVAL_MS);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [autoRefresh, fetchMetrics]);

  const confidencePct = metrics ? Math.round(metrics.avgConfidence * 100) : 0;

  return (
    <div className="space-y-5">
      {/* ── Toolbar ── */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <p className="text-xs text-slate-400">
          Polling every 3 s · Prometheus <code className="bg-slate-100 px-1 rounded">/metrics</code>
        </p>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoRefresh((v) => !v)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              autoRefresh
                ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                : 'bg-slate-50 border-slate-200 text-slate-600'
            }`}
          >
            {autoRefresh
              ? <><Wifi className="w-3.5 h-3.5" /> Live</>
              : <><WifiOff className="w-3.5 h-3.5" /> Paused</>}
          </button>
          <button
            onClick={fetchMetrics}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white text-xs font-semibold rounded-lg transition-colors shadow-sm"
          >
            {loading
              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
              : <RefreshCw className="w-3.5 h-3.5" />}
            Refresh
          </button>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Connection Error</p>
            <p className="mt-0.5 text-red-600">{error}</p>
          </div>
        </div>
      )}

      {/* ── Stat cards + Confidence chart side-by-side ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,2.5fr)] gap-5">
        {/* Left: stat cards stacked */}
        <div className="flex flex-col gap-3">
          <StatCard
            icon={TrendingUp}
            iconBg="bg-indigo-50"
            iconColor="text-indigo-600"
            label="Total Predictions"
            value={metrics ? metrics.totalPredictions.toLocaleString() : '—'}
            sub="model_predictions_total"
          />
          <StatCard
            icon={CheckCircle2}
            iconBg="bg-emerald-50"
            iconColor="text-emerald-600"
            label="Model Confidence"
            value={metrics ? `${confidencePct}%` : '—'}
            sub="model_prediction_confidence_score"
          />
        </div>

        {/* Right: Confidence trend */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1 rounded-md bg-indigo-50">
              <TrendingUp className="w-3.5 h-3.5 text-indigo-600" />
            </div>
            <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
              Confidence Score Trend
            </h3>
            {history.length > 0 && (
              <span className="ml-auto text-xs text-slate-400">{history.length} pts</span>
            )}
          </div>
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={history} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} unit="%" />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 11 }}
                  formatter={(v) => [`${v}%`, 'Confidence']}
                />
                <Area
                  type="monotone"
                  dataKey="avgConfidence"
                  stroke="#6366f1"
                  strokeWidth={2}
                  fill="url(#confGrad)"
                  dot={false}
                  activeDot={{ r: 3, fill: '#6366f1' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-40 flex items-center justify-center text-xs text-slate-400">
              Waiting for data…
            </div>
          )}
        </div>
      </div>

      {/* ── Category distribution + Requests over time ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Category distribution bar chart */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1 rounded-md bg-purple-50">
              <BarChart2 className="w-3.5 h-3.5 text-purple-600" />
            </div>
            <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
              Prediction Category Distribution
            </h3>
          </div>
          {categoryDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={categoryDist} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 11 }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {categoryDist.map((entry) => (
                    <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name] ?? DEFAULT_COLOR} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-44 flex flex-col items-center justify-center text-slate-400 text-xs gap-2">
              <BarChart2 className="w-7 h-7 opacity-30" />
              No category data yet — run some predictions first.
            </div>
          )}
        </div>

        {/* Total requests over time */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1 rounded-md bg-emerald-50">
              <TrendingUp className="w-3.5 h-3.5 text-emerald-600" />
            </div>
            <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
              Total Requests Over Time
            </h3>
          </div>
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={history} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="reqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#10b981" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 11 }}
                  formatter={(v) => [v, 'Requests']}
                />
                <Area
                  type="monotone"
                  dataKey="totalPredictions"
                  stroke="#10b981"
                  strokeWidth={2}
                  fill="url(#reqGrad)"
                  dot={false}
                  activeDot={{ r: 3, fill: '#10b981' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-44 flex flex-col items-center justify-center text-slate-400 text-xs gap-2">
              <TrendingUp className="w-7 h-7 opacity-30" />
              Waiting for data…
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
