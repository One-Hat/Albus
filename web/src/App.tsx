import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  Lock,
  PlayCircle,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import type { RunIndexItem, RunSummary } from '@contracts/types';
import {
  type ApiConfig,
  fetchApiConfig,
  fetchRunIndex,
  fetchRunSummary,
  type LedgerVerifyResult,
  verifyRunLedger,
} from './api/client';
import {
  SourceBadge,
  TaggedNumber,
  VerdictBadge,
} from './components/SourceBadge';
import { EquityCurveChart } from './components/EquityCurveChart';
import { CalibrationPanel } from './components/CalibrationPanel';
import { EvidenceInspector } from './components/EvidenceInspector';
import { StagePipelineModal } from './components/StagePipelineModal';

type ActiveSection = 'overview' | 'backtest' | 'evidence' | 'swarm' | 'ledger';

export const App: React.FC = () => {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [runs, setRuns] = useState<RunIndexItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [currentRun, setCurrentRun] = useState<RunSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveSection>('overview');
  const [ledgerCheck, setLedgerCheck] = useState<LedgerVerifyResult | null>(
    null
  );
  const [verifyingLedger, setVerifyingLedger] = useState<boolean>(false);
  const [pipelineModalOpen, setPipelineModalOpen] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        setLoading(true);
        const [cfg, indexItems] = await Promise.all([
          fetchApiConfig(),
          fetchRunIndex(),
        ]);
        if (cancelled) return;
        setConfig(cfg);
        setRuns(indexItems);

        if (indexItems.length > 0) {
          // Prefer real-groq-run-01 or first item
          const initialId = indexItems[0].id;
          setSelectedRunId(initialId);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    init();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    let cancelled = false;
    async function loadRun() {
      try {
        setLoading(true);
        setError(null);
        setLedgerCheck(null);
        const summary = await fetchRunSummary(selectedRunId);
        if (!cancelled) {
          setCurrentRun(summary);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    loadRun();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  const handleVerifyLedger = async () => {
    if (!currentRun) return;
    setVerifyingLedger(true);
    try {
      const res = await verifyRunLedger(currentRun.id, currentRun);
      setLedgerCheck(res);
    } finally {
      setVerifyingLedger(false);
    }
  };

  if (loading && !currentRun) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-200 flex items-center justify-center p-6">
        <div className="flex items-center gap-3 text-sm font-mono">
          <RefreshCw className="w-5 h-5 text-emerald-400 animate-spin" />
          <span>Loading Albus Quantitative Research Telemetry...</span>
        </div>
      </div>
    );
  }

  if (error || !currentRun) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-200 flex items-center justify-center p-6">
        <div className="max-w-lg w-full bg-rose-950/30 border border-rose-500/40 rounded-xl p-6 space-y-3">
          <div className="flex items-center gap-2 text-rose-300 font-bold">
            <AlertOctagon className="w-5 h-5" />
            <span>Telemetry Load Error</span>
          </div>
          <p className="text-xs font-mono text-rose-200">
            {error || 'No run summary available.'}
          </p>
        </div>
      </div>
    );
  }

  const isMock = currentRun.source === 'MOCK';
  const failedChecks = currentRun.verdict.checks.filter((c) => !c.passed);
  const uniqueTickers = Array.from(new Set(currentRun.universe));

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Sticky Source Mode Banner */}
      {isMock ? (
        <div
          data-testid="mock-warning-banner"
          className="bg-amber-500/20 border-b border-amber-500/50 px-4 py-2.5 text-amber-200 text-xs font-mono flex flex-wrap items-center justify-between gap-2"
        >
          <div className="flex items-center gap-2 font-bold">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              MOCK DATA DEMO — SYNTHETIC FIXTURE ({currentRun.id}). EVERY NUMBER
              ON SCREEN IS LABELED [MOCK] AND MUST NEVER BE INTERPRETED AS REAL.
            </span>
          </div>
          <SourceBadge source="MOCK" size="sm" />
        </div>
      ) : (
        <div
          data-testid="real-mode-banner"
          className="bg-emerald-950/50 border-b border-emerald-500/40 px-4 py-2 text-emerald-200 text-xs font-mono flex flex-wrap items-center justify-between gap-2"
        >
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              REAL VERIFIED RUN ({currentRun.id}) — SEC EDGAR 10-K Filings +{' '}
              {currentRun.provider.toUpperCase()} Open-Source Inference (No
              Lookahead Verified)
            </span>
          </div>
          <SourceBadge source="REAL" size="sm" />
        </div>
      )}

      {/* Top Navigation & Provider Header */}
      <header className="border-b border-slate-800/90 bg-slate-900/60 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-300 font-mono font-bold text-lg">
              A
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-slate-100">
                  ALBUS
                </h1>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  contract v{currentRun.contract_version}
                </span>
                <SourceBadge source={currentRun.source} size="sm" />
              </div>
              <p className="text-xs text-slate-400">
                Quant Research Copilot & Self-Calibrating Forecasting Swarm
              </p>
            </div>
          </div>

          {/* Model Tier Pills & Run Switcher */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="hidden xl:flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-[11px] font-mono text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-indigo-400 mr-1" />
              <span className="text-slate-400">Provider:</span>
              <span className="text-emerald-300 font-semibold uppercase">
                {currentRun.provider}
              </span>
              <span className="text-slate-600">|</span>
              <span title="Small / Extraction Model">
                S: {currentRun.models.small}
              </span>
              <span className="text-slate-600">·</span>
              <span title="Mid / Research & Swarm Model">
                M: {currentRun.models.mid}
              </span>
              <span className="text-slate-600">·</span>
              <span title="Large / Ultra Critic Model">
                L: {currentRun.models.large}
              </span>
            </div>

            {/* Run Switcher Dropdown */}
            <div className="flex items-center gap-2">
              <label
                htmlFor="run-selector"
                className="text-xs font-mono text-slate-400"
              >
                Run:
              </label>
              <select
                id="run-selector"
                aria-label="Select Replay Run"
                value={selectedRunId}
                onChange={(e) => setSelectedRunId(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 focus:outline-none focus:border-emerald-500"
              >
                {runs.map((item) => (
                  <option key={item.id} value={item.id}>
                    [{item.source}] {item.id} — ({item.verdict.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => setPipelineModalOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 text-xs font-mono font-semibold transition-colors"
            >
              <PlayCircle className="w-3.5 h-3.5" />
              <span>Pipeline & Live Runner</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 flex-1 space-y-6">
        {/* Run Explorer Bar (Quick Switch Cards) */}
        <section aria-label="Available Runs">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {runs.map((item) => {
              const isSelected = item.id === currentRun.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setSelectedRunId(item.id)}
                  className={`text-left p-3.5 rounded-xl border transition-all flex items-start justify-between gap-3 ${
                    isSelected
                      ? 'bg-slate-900 border-emerald-500/60 shadow-lg shadow-emerald-950/20'
                      : 'bg-slate-900/40 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-slate-100">
                        {item.id}
                      </span>
                      <SourceBadge source={item.source} size="xs" />
                      <VerdictBadge label={item.verdict} size="sm" />
                    </div>
                    <p className="text-xs text-slate-300 line-clamp-1">
                      {item.hypothesis}
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 shrink-0">
                    {item.created_at.slice(0, 10)}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Caveat State Banner (Triggered when failedChecks > 0 or small sample) */}
        {(failedChecks.length > 0 || currentRun.backtest.n_events < 40) && (
          <section
            data-testid="caveat-banner"
            className="bg-amber-950/30 border border-amber-500/40 rounded-xl p-4 space-y-2"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-amber-300 text-xs font-bold uppercase tracking-wider">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <span>
                  Critic Methodology Caveats Active ({failedChecks.length}{' '}
                  Failed Check{failedChecks.length === 1 ? '' : 's'} · N=
                  {currentRun.backtest.n_events} Events)
                </span>
              </div>
              <SourceBadge source={currentRun.source} size="xs" />
            </div>
            <ul className="text-xs text-amber-100/90 space-y-1 pl-6 list-disc">
              {failedChecks.map((fc) => (
                <li key={fc.name}>
                  <span className="font-semibold">{fc.name}:</span>{' '}
                  {fc.evidence}
                </li>
              ))}
              {currentRun.backtest.n_events < 40 && (
                <li>
                  <span className="font-semibold">Sample Size Notice:</span>{' '}
                  Backtest evaluated{' '}
                  <TaggedNumber
                    value={currentRun.backtest.n_events}
                    source={currentRun.source}
                  />{' '}
                  events ({currentRun.backtest.n_dropped} filings dropped due to
                  missing prior-year baselines). Interpret Sharpe confidence
                  intervals accordingly.
                </li>
              )}
            </ul>
          </section>
        )}

        {/* Hypothesis & Verdict Hero Card */}
        <section className="bg-slate-900/75 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-1.5 max-w-4xl">
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
                <span>RUN ID: {currentRun.id}</span>
                <span>·</span>
                <span>CREATED: {currentRun.created_at}</span>
                <span>·</span>
                <SourceBadge source={currentRun.source} size="sm" />
              </div>
              <h2 className="text-lg sm:text-xl font-bold text-slate-100 leading-snug">
                {currentRun.hypothesis}
              </h2>
            </div>

            <div className="flex flex-col items-end gap-1.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                Ultra Critic Verdict
              </span>
              <VerdictBadge label={currentRun.verdict.label} size="lg" />
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/90 text-xs text-slate-300 leading-relaxed">
            <span className="font-mono uppercase text-[10px] text-indigo-400 block mb-1">
              Adjudicator Narrative ({currentRun.models.large}):
            </span>
            {currentRun.verdict.narrative}
          </div>

          {/* Universe Tickers Strip */}
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-xs font-mono text-slate-400 mr-1">
              Universe (
              <TaggedNumber
                value={currentRun.universe.length}
                source={currentRun.source}
              />{' '}
              evals across{' '}
              <TaggedNumber
                value={uniqueTickers.length}
                source={currentRun.source}
              />{' '}
              tickers):
            </span>
            {uniqueTickers.map((ticker) => (
              <span
                key={ticker}
                className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300"
              >
                {ticker}
              </span>
            ))}
          </div>
        </section>

        {/* Section Navigation Tabs */}
        <nav
          className="flex flex-wrap gap-2 border-b border-slate-800 pb-3"
          aria-label="Dashboard sections"
        >
          {(
            [
              {
                id: 'overview',
                label: 'All Panels (Full View)',
                icon: Activity,
              },
              {
                id: 'backtest',
                label: 'Backtest & Equity Curve',
                icon: TrendingUp,
              },
              {
                id: 'evidence',
                label: `Verbatim Citations (${currentRun.evidence.length})`,
                icon: FileText,
              },
              {
                id: 'swarm',
                label: 'Swarm & Brier Calibration',
                icon: Database,
              },
              {
                id: 'ledger',
                label: 'SHA-256 Ledger & Audit Checks',
                icon: Lock,
              },
            ] as const
          ).map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-mono transition-colors ${
                  active
                    ? 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40 font-semibold'
                    : 'bg-slate-900/60 text-slate-400 border border-slate-800 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* 1. Critic Checks Grid */}
        {(activeTab === 'overview' || activeTab === 'ledger') && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                Deterministic & Critic Gate Checks
              </h3>
              <SourceBadge source={currentRun.source} size="xs" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {currentRun.verdict.checks.map((chk) => (
                <div
                  key={chk.name}
                  className={`p-4 rounded-xl border ${
                    chk.passed
                      ? 'bg-slate-900/70 border-slate-800'
                      : 'bg-amber-950/25 border-amber-500/50'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-xs font-bold text-slate-100">
                      {chk.name}
                    </span>
                    {chk.passed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-amber-400 shrink-0" />
                    )}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {chk.evidence}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* 2. Backtest KPIs, Equity Curve & Dropped Filings Disclosure */}
        {(activeTab === 'overview' || activeTab === 'backtest') && (
          <section className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Net Sharpe
                </div>
                <div className="mt-1 text-base font-bold text-slate-100">
                  <TaggedNumber
                    value={currentRun.backtest.sharpe.toFixed(3)}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  95% Sharpe CI
                </div>
                <div className="mt-1 text-xs font-bold text-slate-200">
                  <TaggedNumber
                    value={`[${currentRun.backtest.sharpe_ci[0].toFixed(
                      2
                    )}, ${currentRun.backtest.sharpe_ci[1].toFixed(2)}]`}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Max Drawdown
                </div>
                <div className="mt-1 text-base font-bold text-rose-300">
                  <TaggedNumber
                    value={`${(currentRun.backtest.max_drawdown * 100).toFixed(
                      2
                    )}%`}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Total Return
                </div>
                <div
                  className={`mt-1 text-base font-bold ${
                    currentRun.backtest.total_return >= 0
                      ? 'text-emerald-300'
                      : 'text-rose-300'
                  }`}
                >
                  <TaggedNumber
                    value={`${
                      currentRun.backtest.total_return >= 0 ? '+' : ''
                    }${(currentRun.backtest.total_return * 100).toFixed(2)}%`}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Turnover
                </div>
                <div className="mt-1 text-base font-bold text-slate-100">
                  <TaggedNumber
                    value={`${currentRun.backtest.turnover.toFixed(2)}x`}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Cost Drag (10bps)
                </div>
                <div className="mt-1 text-base font-bold text-amber-300">
                  <TaggedNumber
                    value={`${(currentRun.backtest.cost_drag * 100).toFixed(
                      2
                    )}%`}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Events Traded
                </div>
                <div className="mt-1 text-base font-bold text-slate-100">
                  <TaggedNumber
                    value={currentRun.backtest.n_events}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Variants Tested (k)
                </div>
                <div className="mt-1 text-base font-bold text-slate-100">
                  <TaggedNumber
                    value={currentRun.backtest.k_variants}
                    source={currentRun.source}
                  />
                </div>
              </div>
            </div>

            <EquityCurveChart
              backtest={currentRun.backtest}
              source={currentRun.source}
            />

            {/* Dropped Filings Audit Card */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Dropped Filings Audit Reconciliation
                  </h4>
                  <SourceBadge source={currentRun.source} size="xs" />
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Total Dropped:{' '}
                  <TaggedNumber
                    value={currentRun.backtest.n_dropped}
                    source={currentRun.source}
                    className="text-slate-200 font-semibold"
                  />{' '}
                  filings excluded prior to trade execution.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {Object.entries(currentRun.backtest.dropped_reasons).map(
                  ([reason, count]) => (
                    <div
                      key={reason}
                      className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono flex items-center gap-2"
                    >
                      <span className="text-slate-400">{reason}:</span>
                      <TaggedNumber
                        value={count}
                        source={currentRun.source}
                        className="text-amber-300 font-bold"
                      />
                    </div>
                  )
                )}
              </div>
            </div>
          </section>
        )}

        {/* 3. Verbatim SEC Filing Evidence & Citation Inspector */}
        {(activeTab === 'overview' || activeTab === 'evidence') && (
          <section>
            <EvidenceInspector
              evidence={currentRun.evidence}
              stats={currentRun.stats}
              source={currentRun.source}
            />
          </section>
        )}

        {/* 4. Self-Calibrating Swarm & Brier Panel */}
        {(activeTab === 'overview' || activeTab === 'swarm') && (
          <section>
            <CalibrationPanel
              calibrations={currentRun.calibration}
              swarm={currentRun.swarm}
              source={currentRun.source}
            />
          </section>
        )}

        {/* 5. SHA-256 Hash-Chained Ledger Card */}
        {(activeTab === 'overview' || activeTab === 'ledger') && (
          <section className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <Lock className="w-4 h-4 text-emerald-400" />
                  <h2 className="text-base font-semibold text-slate-100">
                    Tamper-Evident SHA-256 Forecast & Evidence Ledger
                  </h2>
                  <SourceBadge source={currentRun.source} size="sm" />
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Every forecast and citation is chained from GENESIS via
                  SHA-256 before outcomes resolve.
                </p>
              </div>

              <button
                onClick={handleVerifyLedger}
                disabled={verifyingLedger}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-200 border border-emerald-500/40 text-xs font-mono font-semibold transition-colors"
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 ${
                    verifyingLedger ? 'animate-spin' : ''
                  }`}
                />
                <span>Verify Hash Chain Now</span>
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Chain Length (Blocks)
                </div>
                <div className="mt-1 text-lg font-bold text-slate-100">
                  <TaggedNumber
                    value={currentRun.ledger.length}
                    source={currentRun.source}
                  />
                </div>
              </div>

              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Integrity Status
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span
                    className={`text-sm font-mono font-bold ${
                      currentRun.ledger.verified
                        ? 'text-emerald-300'
                        : 'text-rose-300'
                    }`}
                  >
                    {currentRun.ledger.verified
                      ? 'VERIFIED_INTACT'
                      : 'TAMPER_DETECTED'}
                  </span>
                  <SourceBadge source={currentRun.source} size="xs" />
                </div>
              </div>

              <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">
                  Head SHA-256 Digest
                </div>
                <div className="mt-1 font-mono text-xs text-emerald-300 break-all">
                  {currentRun.ledger.head_hash}
                </div>
              </div>
            </div>

            {ledgerCheck && (
              <div
                className={`p-3.5 rounded-xl border text-xs font-mono flex flex-wrap items-center justify-between gap-2 ${
                  ledgerCheck.verified
                    ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                    : 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                }`}
              >
                <div>
                  <span className="font-bold">
                    {ledgerCheck.verified
                      ? '[OK] Cryptographic Audit Passed:'
                      : '[FAIL] Ledger Broken:'}
                  </span>{' '}
                  Recomputed SHA-256 chain across{' '}
                  <TaggedNumber
                    value={ledgerCheck.length}
                    source={currentRun.source}
                  />{' '}
                  blocks matches head hash ({ledgerCheck.head_hash.slice(0, 16)}
                  ...).
                </div>
                <SourceBadge source={currentRun.source} size="xs" />
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer Disclaimer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-5 px-4 sm:px-6 mt-10">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
          <div>
            <span className="font-semibold text-slate-400">
              NOT INVESTMENT ADVICE:
            </span>{' '}
            Albus is an empirical quantitative research & forecasting
            calibration system. Past backtest performance and Brier calibration
            do not guarantee future returns.
          </div>
          <div className="flex items-center gap-3 font-mono">
            <span>Contract v{currentRun.contract_version}</span>
            <span>·</span>
            <SourceBadge source={currentRun.source} size="xs" />
          </div>
        </div>
      </footer>

      {config && (
        <StagePipelineModal
          isOpen={pipelineModalOpen}
          onClose={() => setPipelineModalOpen(false)}
          config={config}
          currentRun={currentRun}
        />
      )}
    </div>
  );
};

export default App;
