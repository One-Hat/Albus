import React, { useState } from 'react';
import type { Calibration, Source, Swarm } from '@contracts/types';
import { SourceBadge, TaggedNumber } from './SourceBadge';

interface CalibrationPanelProps {
  calibrations: Calibration[];
  swarm: Swarm;
  source: Source;
}

const MODEL_LABELS: Record<string, string> = {
  ensemble: 'Calibrated Logit Ensemble (Swarm)',
  equal_weight_swarm: 'Equal-Weight Swarm',
  single_large: 'Single Large Model (Ultra)',
  single_mid: 'Single Mid Model (Super)',
  base_rate: 'Historical Base Rate',
};

export const CalibrationPanel: React.FC<CalibrationPanelProps> = ({
  calibrations,
  swarm,
  source,
}) => {
  const [selectedKind, setSelectedKind] = useState<'retrospective_clean' | 'live'>(
    calibrations[0]?.kind || 'retrospective_clean'
  );

  const activeCal =
    calibrations.find((c) => c.kind === selectedKind) || calibrations[0];

  const brierEntries = activeCal
    ? Object.entries(activeCal.brier).sort((a, b) => a[1] - b[1])
    : [];
  const maxBrier = Math.max(...brierEntries.map(([, v]) => v), 0.25);

  // Reliability plot coordinates
  const svgSize = 260;
  const pad = 34;
  const inner = svgSize - pad * 2;
  const toX = (val: number) => pad + val * inner;
  const toY = (val: number) => pad + inner - val * inner;

  const agentNames =
    swarm.weights_over_time.length > 0
      ? Object.keys(swarm.weights_over_time[0].weights)
      : [];

  return (
    <div className="space-y-6">
      {/* Header & Strict Kind Separation Selector */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-slate-100">
                Self-Calibrating Forecasting Swarm & Brier Telemetry
              </h2>
              <SourceBadge source={source} size="sm" />
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Strict Kind Separation Rule: Retrospective Clean and Live Forward
              resolutions are evaluated in isolated buckets and never pooled.
            </p>
          </div>

          <div
            className="inline-flex rounded-lg bg-slate-950 p-1 border border-slate-800"
            role="tablist"
            aria-label="Calibration kind tabs"
          >
            {(['retrospective_clean', 'live'] as const).map((kind) => {
              const entry = calibrations.find((c) => c.kind === kind);
              const isSelected = activeCal?.kind === kind;
              return (
                <button
                  key={kind}
                  role="tab"
                  aria-selected={isSelected}
                  onClick={() => setSelectedKind(kind)}
                  disabled={!entry}
                  className={`px-3 py-1.5 rounded-md text-xs font-mono transition-colors flex items-center gap-2 ${
                    isSelected
                      ? 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  } ${!entry ? 'opacity-40 cursor-not-allowed' : ''}`}
                >
                  <span>
                    {kind === 'retrospective_clean'
                      ? 'RETROSPECTIVE_CLEAN'
                      : 'LIVE_RESOLVED'}
                  </span>
                  {entry && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-900 text-slate-300">
                      N={entry.n_resolved}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {activeCal ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-5">
            {/* Left 7 cols: Brier Score Comparison with Bootstrap 95% CIs */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  Brier Score Benchmark (Lower is Better)
                </h3>
                <div className="flex items-center gap-4 text-xs text-slate-400">
                  <span>
                    Resolved Questions:{' '}
                    <TaggedNumber
                      value={activeCal.n_resolved}
                      source={source}
                      className="text-slate-200 font-semibold"
                    />
                  </span>
                  <span>
                    Expected Calibration Error (ECE):{' '}
                    <TaggedNumber
                      value={activeCal.ece.toFixed(4)}
                      source={source}
                      className="text-slate-200 font-semibold"
                    />
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                {brierEntries.map(([key, score], idx) => {
                  const ci = activeCal.brier_ci[key];
                  const widthPct = Math.min(100, Math.max(6, (score / maxBrier) * 100));
                  const isBest = idx === 0;
                  return (
                    <div
                      key={key}
                      className={`p-3 rounded-lg border ${
                        isBest
                          ? 'bg-emerald-950/25 border-emerald-500/40'
                          : 'bg-slate-950/60 border-slate-800/80'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-1.5">
                        <span className="font-medium text-slate-200 flex items-center gap-2">
                          {MODEL_LABELS[key] || key}
                          {isBest && (
                            <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                              Best Calibrated
                            </span>
                          )}
                        </span>
                        <div className="flex items-center gap-3">
                          {ci && (
                            <span className="text-[11px] text-slate-400 font-mono">
                              95% CI:{' '}
                              <TaggedNumber
                                value={`[${ci[0].toFixed(4)}, ${ci[1].toFixed(4)}]`}
                                source={source}
                              />
                            </span>
                          )}
                          <TaggedNumber
                            value={score.toFixed(4)}
                            source={source}
                            className="text-sm font-bold text-slate-100"
                          />
                        </div>
                      </div>
                      <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isBest ? 'bg-emerald-400' : 'bg-slate-600'
                          }`}
                          style={{ width: `${widthPct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right 5 cols: SVG Reliability Diagram + Bin Breakdown */}
            <div className="lg:col-span-5 bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Reliability Diagram ({activeCal.kind})
                  </h3>
                  <SourceBadge source={source} size="xs" />
                </div>

                <svg
                  viewBox={`0 0 ${svgSize} ${svgSize}`}
                  className="w-full max-w-[250px] mx-auto h-auto select-none"
                  role="img"
                  aria-label="Calibration reliability diagram"
                >
                  {/* Grid box */}
                  <rect
                    x={pad}
                    y={pad}
                    width={inner}
                    height={inner}
                    fill="#020617"
                    stroke="#1e293b"
                  />
                  {/* Perfect calibration diagonal y=x */}
                  <line
                    x1={toX(0)}
                    y1={toY(0)}
                    x2={toX(1)}
                    y2={toY(1)}
                    stroke="#64748b"
                    strokeDasharray="4 4"
                    strokeWidth="1.2"
                  />
                  {/* Calibration curve */}
                  {activeCal.reliability.length > 1 && (
                    <polyline
                      fill="none"
                      stroke={source === 'MOCK' ? '#f59e0b' : '#10b981'}
                      strokeWidth="2.2"
                      points={activeCal.reliability
                        .map(
                          (b) =>
                            `${toX(b.mean_pred).toFixed(1)},${toY(b.freq).toFixed(1)}`
                        )
                        .join(' ')}
                    />
                  )}
                  {/* Points */}
                  {activeCal.reliability.map((b, i) => (
                    <g key={i}>
                      <circle
                        cx={toX(b.mean_pred)}
                        cy={toY(b.freq)}
                        r={4.5}
                        fill={source === 'MOCK' ? '#fbbf24' : '#34d399'}
                        stroke="#020617"
                        strokeWidth="1.5"
                      />
                    </g>
                  ))}
                  {/* Axis labels */}
                  <text
                    x={svgSize / 2}
                    y={svgSize - 6}
                    textAnchor="middle"
                    className="fill-slate-400 text-[9px] font-mono"
                  >
                    Mean Predicted Probability
                  </text>
                  <text
                    x={10}
                    y={svgSize / 2}
                    textAnchor="middle"
                    transform={`rotate(-90 10 ${svgSize / 2})`}
                    className="fill-slate-400 text-[9px] font-mono"
                  >
                    Observed Frequency
                  </text>
                </svg>
              </div>

              {/* Reliability Bins Table */}
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-[11px] font-mono text-left">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-800">
                      <th className="py-1">Bin</th>
                      <th className="py-1">Pred</th>
                      <th className="py-1">Freq</th>
                      <th className="py-1 text-right">Count (n)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {activeCal.reliability.map((bin, idx) => (
                      <tr key={idx} className="text-slate-300">
                        <td className="py-1">
                          [{bin.lo.toFixed(2)}, {bin.hi.toFixed(2)}]
                        </td>
                        <td className="py-1">
                          <TaggedNumber
                            value={bin.mean_pred.toFixed(2)}
                            source={source}
                          />
                        </td>
                        <td className="py-1">
                          <TaggedNumber
                            value={bin.freq.toFixed(2)}
                            source={source}
                          />
                        </td>
                        <td className="py-1 text-right">
                          <TaggedNumber value={bin.n} source={source} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Swarm Agent Weights Evolution Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Swarm Persona Weight Convergence Over Rounds
              </h3>
              <SourceBadge source={source} size="sm" />
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Multiplicative Brier skill re-weighting across{' '}
              <TaggedNumber value={swarm.n_agents} source={source} /> specialist
              personas and{' '}
              <TaggedNumber value={swarm.n_forecasts} source={source} /> total
              forecasts.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 pr-4">Round</th>
                {agentNames.map((agent) => (
                  <th key={agent} className="py-2 px-3 uppercase">
                    {agent}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {swarm.weights_over_time.map((row) => (
                <tr key={row.round} className="hover:bg-slate-800/30">
                  <td className="py-2.5 pr-4 font-semibold text-slate-200">
                    Round{' '}
                    <TaggedNumber value={row.round} source={source} />
                  </td>
                  {agentNames.map((agent) => {
                    const w = row.weights[agent] ?? 0;
                    return (
                      <td key={agent} className="py-2.5 px-3">
                        <div className="flex flex-col gap-1">
                          <TaggedNumber
                            value={`${(w * 100).toFixed(1)}%`}
                            source={source}
                            className="text-slate-200"
                          />
                          <div className="w-24 h-1.5 bg-slate-950 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-400 rounded-full"
                              style={{ width: `${Math.min(100, w * 100 * 2)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
