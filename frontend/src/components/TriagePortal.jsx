import { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Globe,
  Layers,
  Loader2,
  Network,
  Search,
  Send,
  Shield,
  Zap,
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const CATEGORY_META = {
  Database:       { icon: Database,  color: 'bg-blue-100 text-blue-700 border-blue-200' },
  Network:        { icon: Network,   color: 'bg-purple-100 text-purple-700 border-purple-200' },
  Infrastructure: { icon: Cpu,       color: 'bg-orange-100 text-orange-700 border-orange-200' },
  Application:    { icon: Layers,    color: 'bg-green-100 text-green-700 border-green-200' },
  Security:       { icon: Shield,    color: 'bg-red-100 text-red-700 border-red-200' },
};

const PRIORITY_META = {
  high:     'bg-red-100 text-red-700 border-red-200',
  medium:   'bg-yellow-100 text-yellow-700 border-yellow-200',
  low:      'bg-green-100 text-green-700 border-green-200',
  critical: 'bg-rose-100 text-rose-800 border-rose-200',
};

function ConfidenceBar({ value, label }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-yellow-400' : 'bg-red-400';
  return (
    <div>
      <div className="flex justify-between mb-1 text-xs text-slate-500">
        <span>{label}</span>
        <span className="font-semibold text-slate-700">{pct}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function SimilarIncidentCard({ incident }) {
  const similarity = Math.round(incident.similarity_score * 100);
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="text-xs font-mono font-semibold text-slate-400">{incident.incident_id}</span>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 border border-indigo-100 shrink-0">
          {similarity}%
        </span>
      </div>
      <p className="text-xs font-medium text-slate-700 mb-1 leading-snug">{incident.short_description}</p>
      <span className="inline-block text-xs px-2 py-0.5 rounded-full bg-white border border-slate-200 text-slate-500 mb-2">
        {incident.category}
      </span>
      {incident.resolution_notes && (
        <p className="text-xs text-slate-400 leading-relaxed border-t border-slate-200 pt-1.5">
          <span className="font-medium text-slate-500">Fix: </span>
          {incident.resolution_notes}
        </p>
      )}
    </div>
  );
}

/* ─── Empty-state placeholder shown before the first prediction ─── */
function EmptyResults() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 text-center px-6 py-10">
      <div className="w-12 h-12 rounded-2xl bg-indigo-50 flex items-center justify-center mb-4">
        <Zap className="w-6 h-6 text-indigo-400" />
      </div>
      <p className="text-sm font-medium text-slate-500">No prediction yet</p>
      <p className="text-xs text-slate-400 mt-1 max-w-xs">
        Fill in the form on the left and click <strong>Run Triage Prediction</strong> to see AI-generated results here.
      </p>
    </div>
  );
}

export default function TriagePortal() {
  const [form, setForm] = useState({
    short_description: '',
    description: '',
    environment: 'production',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);
  const [incidentCounter, setIncidentCounter] = useState(101);

  const handleChange = (e) =>
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    const payload = {
      incident_id: `INC-${incidentCounter}`,
      short_description: form.short_description,
      description: form.description,
      environment: form.environment,
    };

    try {
      const res = await fetch(`${API_BASE}/api/v1/incidents/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server returned ${res.status}: ${text}`);
      }
      const data = await res.json();
      setResult(data);
      setIncidentCounter((c) => c + 1);
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError(
          'Cannot connect to the backend at http://localhost:8000. Make sure the FastAPI server is running.',
        );
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const CategoryIcon  = result ? (CATEGORY_META[result.predicted_category]?.icon ?? Globe) : null;
  const categoryColor = result
    ? (CATEGORY_META[result.predicted_category]?.color ?? 'bg-slate-100 text-slate-700 border-slate-200')
    : '';
  const priorityColor = result
    ? (PRIORITY_META[result.predicted_priority?.toLowerCase()] ?? 'bg-slate-100 text-slate-700 border-slate-200')
    : '';

  return (
    /* 2-column grid: form | results */
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-5 items-start">

      {/* ── LEFT: Input Form ── */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden lg:sticky lg:top-20">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-indigo-50">
            <Zap className="w-4 h-4 text-indigo-600" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-800">New Incident Triage</h2>
            <p className="text-xs text-slate-400">Submit for AI-powered analysis</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {/* Short description */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5 uppercase tracking-wide">
              Short Description <span className="text-red-400 normal-case">*</span>
            </label>
            <input
              type="text"
              name="short_description"
              value={form.short_description}
              onChange={handleChange}
              required
              placeholder="e.g. High latency on API gateway"
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
            />
          </div>

          {/* Detailed description */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5 uppercase tracking-wide">
              Detailed Description <span className="text-red-400 normal-case">*</span>
            </label>
            <textarea
              name="description"
              value={form.description}
              onChange={handleChange}
              required
              rows={5}
              placeholder="e.g. Gateway returning 504 timeouts under load. Observed spike in error rate…"
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition resize-none"
            />
          </div>

          {/* Environment */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5 uppercase tracking-wide">
              Environment
            </label>
            <select
              name="environment"
              value={form.environment}
              onChange={handleChange}
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
            >
              <option value="production">Production</option>
              <option value="staging">Staging</option>
              <option value="development">Development</option>
              <option value="qa">QA</option>
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors shadow-sm"
          >
            {loading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Analysing…</>
            ) : (
              <><Send className="w-4 h-4" /> Run Triage Prediction</>
            )}
          </button>
        </form>

        {/* Error banner inside the form card */}
        {error && (
          <div className="mx-5 mb-5 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">Connection Error</p>
              <p className="mt-0.5 text-red-600 leading-relaxed">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* ── RIGHT: Results Panel ── */}
      <div className="space-y-4">
        {!result && !loading && <EmptyResults />}

        {loading && (
          <div className="flex flex-col items-center justify-center min-h-64 gap-3 text-slate-400">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
            <p className="text-sm">Running AI triage analysis…</p>
          </div>
        )}

        {result && (
          <>
            {/* ── Summary row: 3 stat cards ── */}
            <div className="grid grid-cols-3 gap-3">
              {/* Category */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-2.5">
                <p className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  <Search className="w-3 h-3" /> Category
                </p>
                <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold ${categoryColor}`}>
                  {CategoryIcon && <CategoryIcon className="w-3.5 h-3.5" />}
                  {result.predicted_category}
                </div>
                <ConfidenceBar value={result.category_confidence} label="Confidence" />
              </div>

              {/* Priority */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-2.5">
                <p className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  <AlertTriangle className="w-3 h-3" /> Priority
                </p>
                <div className={`inline-flex items-center px-2.5 py-1 rounded-lg border text-xs font-semibold capitalize ${priorityColor}`}>
                  {result.predicted_priority}
                </div>
                <ConfidenceBar value={result.priority_confidence} label="Confidence" />
              </div>

              {/* Resolution time */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-2.5">
                <p className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  <Clock className="w-3 h-3" /> Est. Resolution
                </p>
                <p className="text-2xl font-bold text-slate-800 leading-none">
                  {Math.round(result.estimated_resolution_time_minutes)}
                  <span className="text-xs font-medium text-slate-400 ml-1">min</span>
                </p>
                <p className="text-xs text-slate-400 font-mono truncate">
                  {result.trace_id?.slice(0, 12)}…
                </p>
              </div>
            </div>

            {/* ── Troubleshooting ── */}
            {result.troubleshooting_recommendations?.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-1 rounded-md bg-emerald-50">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                  </div>
                  <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Troubleshooting Recommendations
                  </h3>
                </div>
                <ol className="space-y-1.5">
                  {result.troubleshooting_recommendations.map((step, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-xs text-slate-600">
                      <span className="mt-0.5 shrink-0 w-4 h-4 rounded-full bg-indigo-100 text-indigo-600 text-xs font-bold flex items-center justify-center">
                        {i + 1}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* ── Similar Incidents ── */}
            {result.similar_incidents?.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-1 rounded-md bg-indigo-50">
                    <Layers className="w-3.5 h-3.5 text-indigo-600" />
                  </div>
                  <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
                    Similar Historical Incidents
                  </h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                  {result.similar_incidents.map((inc) => (
                    <SimilarIncidentCard key={inc.incident_id} incident={inc} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
