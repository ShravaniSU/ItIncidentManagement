import { Activity, Cpu } from 'lucide-react';
import TriagePortal from './components/TriagePortal';
import ObservabilityDashboard from './components/ObservabilityDashboard';
import './index.css';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Slim brand header — no tabs */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-30">
        <div className="max-w-screen-xl mx-auto px-6 py-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-bold text-slate-800">AI IncidentOps</p>
            <p className="text-xs text-slate-400">MLOps Triage &amp; Observability</p>
          </div>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
            <Activity className="w-3.5 h-3.5 text-emerald-500" />
            Live monitoring active
          </div>
        </div>
      </header>

      {/* Single unified page */}
      <main className="max-w-screen-xl mx-auto px-6 py-6 space-y-6">
        {/* ── Section 1: Incident Triage ── */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <span className="w-1 h-5 rounded-full bg-indigo-600 inline-block" />
            <h1 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
              Incident Triage Portal
            </h1>
          </div>
          <TriagePortal />
        </section>

        {/* ── Divider ── */}
        <div className="border-t border-slate-200" />

        {/* ── Section 2: Observability ── */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <span className="w-1 h-5 rounded-full bg-emerald-500 inline-block" />
            <h1 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
              Observability &amp; Drift Dashboard
            </h1>
          </div>
          <ObservabilityDashboard />
        </section>
      </main>
    </div>
  );
}
