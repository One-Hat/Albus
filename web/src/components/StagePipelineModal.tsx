import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Play,
  ShieldAlert,
  X,
} from 'lucide-react';
import type { RunSummary, StageEvent } from '@contracts/types';
import {
  type ApiConfig,
  replayStageEvents,
  submitLiveRun,
} from '../api/client';
import { SourceBadge } from './SourceBadge';

interface StagePipelineModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ApiConfig;
  currentRun: RunSummary;
}

const STAGES: StageEvent['stage'][] = [
  'ingest',
  'extract',
  'research',
  'swarm',
  'critic',
  'report',
];

const STAGE_TITLES: Record<StageEvent['stage'], string> = {
  ingest: '1. SEC EDGAR 10-K Ingestion & Acceptance Audit',
  extract: '2. Item 1A Risk Factor Extraction & Quote Verification',
  research: '3. Semantic Diff Synthesis & Next-Open Backtest',
  swarm: '4. Multi-Persona Forecasting Swarm & Logit Calibration',
  critic: '5. Ultra Critic Deterministic & Methodology Audit',
  report: '6. SHA-256 Hash-Chained Ledger & Report Sealing',
};

const DEFAULT_TICKERS = [
  'AAPL',
  'MSFT',
  'NVDA',
  'AMZN',
  'GOOGL',
  'META',
  'TSLA',
  'JNJ',
  'V',
  'WMT',
];

export const StagePipelineModal: React.FC<StagePipelineModalProps> = ({
  isOpen,
  onClose,
  config,
  currentRun,
}) => {
  const [hypothesis, setHypothesis] = useState(
    'Companies expanding cybersecurity and AI infrastructure risk disclosures in Item 1A experience post-filing negative drift over 60 trading days.'
  );
  const [tickersInput, setTickersInput] = useState(DEFAULT_TICKERS.join(', '));
  const [eventsByStage, setEventsByStage] = useState<
    Partial<Record<StageEvent['stage'], StageEvent>>
  >({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setIsStreaming(false);
      setErrorMsg(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleReplayTrace = () => {
    setErrorMsg(null);
    setEventsByStage({});
    setIsStreaming(true);

    replayStageEvents(
      currentRun,
      (ev) => {
        setEventsByStage((prev) => ({
          ...prev,
          [ev.stage]: ev,
        }));
      },
      () => {
        setIsStreaming(false);
      }
    );
  };

  const handleLiveSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setEventsByStage({});
    const parsedTickers = tickersInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

    try {
      setIsStreaming(true);
      const { run_id } = await submitLiveRun(hypothesis, parsedTickers);
      // Connect SSE stream
      const evtSource = new EventSource(`/api/run/${encodeURIComponent(run_id)}/events`);
      evtSource.onmessage = (msg) => {
        try {
          const ev: StageEvent = JSON.parse(msg.data);
          setEventsByStage((prev) => ({ ...prev, [ev.stage]: ev }));
          if (ev.stage === 'report' && ev.status === 'done') {
            evtSource.close();
            setIsStreaming(false);
          }
        } catch {
          // ignore parse error
        }
      };
      evtSource.onerror = () => {
        evtSource.close();
        setIsStreaming(false);
      };
    } catch (err) {
      setIsStreaming(false);
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <Activity className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-bold text-slate-100">
              Pipeline Stage Progression & Live Hypothesis Runner
            </h2>
            <SourceBadge source={currentRun.source} size="sm" />
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 p-1 rounded-lg"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Guard Status Banner */}
          <div
            className={`p-4 rounded-xl border text-xs flex items-start gap-3 ${
              config.live_enabled
                ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                : 'bg-amber-950/30 border-amber-500/40 text-amber-200'
            }`}
          >
            <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold">
                {config.live_enabled
                  ? `LIVE EXECUTION ENABLED — Provider: ${config.provider.toUpperCase()}`
                  : 'LIVE EXECUTION GUARDED (LIVE_ENABLED=false by default)'}
              </div>
              <p className="text-slate-300 leading-relaxed">
                {config.live_enabled
                  ? 'POST /api/run is active with IP rate limiting, ticker allowlist validation, and strict token budget caps.'
                  : 'To protect API credits on public deployments, live LLM swarm runs are locked behind LIVE_ENABLED=false. You can inspect the 6-stage SSE pipeline trace for the active replay or test the live API guard below.'}
              </p>
              <div className="flex flex-wrap gap-3 pt-1 font-mono text-[11px] text-slate-400">
                <span>Max Tickers: {config.limits.max_tickers}</span>
                <span>Max Tokens/Call: {config.limits.max_tokens_per_call}</span>
                <span>Wall Cap: {config.limits.max_wall_time_sec}s</span>
                <span>Rate Limit: {config.limits.rate_limit_rpm} req/min</span>
              </div>
            </div>
          </div>

          {/* Hypothesis Form */}
          <form onSubmit={handleLiveSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Quantitative Hypothesis (Untrusted Input Sanitized)
              </label>
              <textarea
                rows={2}
                value={hypothesis}
                onChange={(e) => setHypothesis(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Universe Tickers (Allowlist Verified, Max 20)
              </label>
              <input
                type="text"
                value={tickersInput}
                onChange={(e) => setTickersInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 focus:outline-none focus:border-emerald-500"
              />
            </div>

            {errorMsg && (
              <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-500/40 text-rose-200 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{errorMsg}</span>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleReplayTrace}
                disabled={isStreaming}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs transition-colors disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Stream 6-Stage Pipeline Trace ({currentRun.id})</span>
              </button>

              <button
                type="submit"
                disabled={isStreaming}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold text-xs transition-colors disabled:opacity-50"
              >
                <span>Trigger POST /api/run</span>
              </button>
            </div>
          </form>

          {/* 6-Stage SSE Pipeline Progress */}
          <div className="space-y-2.5 pt-2 border-t border-slate-800">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              StageEvent Stream (SSE Contract)
            </h3>
            {STAGES.map((stage) => {
              const ev = eventsByStage[stage];
              const status = ev?.status;
              return (
                <div
                  key={stage}
                  className={`p-3 rounded-xl border text-xs flex items-center justify-between gap-3 ${
                    status === 'done'
                      ? 'bg-emerald-950/20 border-emerald-500/30 text-slate-100'
                      : status === 'start' || status === 'progress'
                      ? 'bg-indigo-950/30 border-indigo-500/40 text-slate-100'
                      : 'bg-slate-950/60 border-slate-800/80 text-slate-400'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {status === 'done' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : status === 'start' || status === 'progress' ? (
                      <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />
                    ) : (
                      <span className="w-4 h-4 rounded-full border border-slate-700 inline-block shrink-0" />
                    )}
                    <div>
                      <div className="font-semibold">{STAGE_TITLES[stage]}</div>
                      {ev?.message && (
                        <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                          {ev.message}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-[11px]">
                    {ev?.done !== undefined && ev?.total !== undefined && (
                      <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300">
                        {ev.done}/{ev.total}
                      </span>
                    )}
                    <span className="uppercase px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                      {status || 'idle'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
