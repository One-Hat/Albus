import React, { useState } from 'react';
import type { Backtest, Source } from '@contracts/types';
import { SourceBadge, TaggedNumber } from './SourceBadge';

interface EquityCurveChartProps {
  backtest: Backtest;
  source: Source;
}

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  backtest,
  source,
}) => {
  const curve = backtest.equity_curve;
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  if (!curve || curve.length === 0) {
    return (
      <div className="p-6 text-center text-sm text-slate-400 border border-slate-800 rounded-lg">
        No equity curve points recorded for this run.
      </div>
    );
  }

  const width = 760;
  const height = 250;
  const padLeft = 56;
  const padRight = 24;
  const padTop = 24;
  const padBottom = 38;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const values = curve.map((p) => p.value);
  const rawMin = Math.min(...values, 0.95);
  const rawMax = Math.max(...values, 1.05);
  const minVal = rawMin - (rawMax - rawMin) * 0.08;
  const maxVal = rawMax + (rawMax - rawMin) * 0.08;

  const toX = (i: number) =>
    padLeft + (curve.length === 1 ? plotW / 2 : (i / (curve.length - 1)) * plotW);
  const toY = (v: number) =>
    padTop + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH;

  const pointsStr = curve
    .map((pt, i) => `${toX(i).toFixed(1)},${toY(pt.value).toFixed(1)}`)
    .join(' ');

  const areaStr = `${toX(0).toFixed(1)},${(padTop + plotH).toFixed(1)} ${pointsStr} ${toX(
    curve.length - 1
  ).toFixed(1)},${(padTop + plotH).toFixed(1)}`;

  // Find peak and trough
  let peakIdx = 0;
  let troughIdx = 0;
  curve.forEach((pt, idx) => {
    if (pt.value > curve[peakIdx].value) peakIdx = idx;
    if (pt.value < curve[troughIdx].value) troughIdx = idx;
  });

  const activePoint = hoverIdx !== null ? curve[hoverIdx] : curve[curve.length - 1];
  const isMock = source === 'MOCK';
  const strokeColor = isMock ? '#f59e0b' : '#10b981';

  const baselineY = toY(1.0);
  const yTicks = [minVal, (minVal + maxVal) / 2, maxVal];

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Cumulative Equity Curve (Next-Open Execution + 10bps Cost)
            </h3>
            <SourceBadge source={source} size="sm" />
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Peak ({curve[peakIdx].date}):{' '}
            <TaggedNumber
              value={`${curve[peakIdx].value.toFixed(3)}x`}
              source={source}
            />{' '}
            · Trough ({curve[troughIdx].date}):{' '}
            <TaggedNumber
              value={`${curve[troughIdx].value.toFixed(3)}x`}
              source={source}
            />
          </p>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
          <span className="text-slate-400 mr-2">{activePoint.date}:</span>
          <TaggedNumber
            value={`${activePoint.value.toFixed(3)}x (${
              activePoint.value >= 1
                ? `+${((activePoint.value - 1) * 100).toFixed(1)}%`
                : `${((activePoint.value - 1) * 100).toFixed(1)}%`
            })`}
            source={source}
            className="text-slate-100 font-semibold"
          />
        </div>
      </div>

      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto select-none"
          role="img"
          aria-label="Backtest cumulative equity curve"
        >
          <defs>
            <linearGradient id={`eq-grad-${source}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={strokeColor} stopOpacity="0.28" />
              <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Horizontal Y gridlines */}
          {yTicks.map((tick, i) => {
            const y = toY(tick);
            return (
              <g key={i}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="3 3"
                />
                <text
                  x={padLeft - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-slate-500 text-[10px] font-mono"
                >
                  {tick.toFixed(2)}x
                </text>
              </g>
            );
          })}

          {/* Baseline 1.00x */}
          {baselineY >= padTop && baselineY <= padTop + plotH && (
            <line
              x1={padLeft}
              y1={baselineY}
              x2={width - padRight}
              y2={baselineY}
              stroke="#475569"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
          )}

          {/* Shaded area & line */}
          <polygon points={areaStr} fill={`url(#eq-grad-${source})`} />
          <polyline
            fill="none"
            stroke={strokeColor}
            strokeWidth="2.5"
            points={pointsStr}
          />

          {/* Data points & hover targets */}
          {curve.map((pt, i) => {
            const cx = toX(i);
            const cy = toY(pt.value);
            const isHovered = hoverIdx === i;
            return (
              <g
                key={pt.date}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx(null)}
                className="cursor-pointer"
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 5.5 : 3.5}
                  fill={isHovered ? '#f8fafc' : strokeColor}
                  stroke="#020617"
                  strokeWidth="1.5"
                />
                <rect
                  x={cx - plotW / (curve.length * 2)}
                  y={padTop}
                  width={plotW / curve.length}
                  height={plotH}
                  fill="transparent"
                />
                {(i === 0 ||
                  i === curve.length - 1 ||
                  i % Math.max(1, Math.floor(curve.length / 5)) === 0) && (
                  <text
                    x={cx}
                    y={height - 10}
                    textAnchor="middle"
                    className="fill-slate-400 text-[10px] font-mono"
                  >
                    {pt.date}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
