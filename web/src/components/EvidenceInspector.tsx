import React, { useState } from 'react';
import { ExternalLink, FileCheck2, Search } from 'lucide-react';
import type { Evidence, PipelineStats, Source } from '@contracts/types';
import { SourceBadge, TaggedNumber } from './SourceBadge';

interface EvidenceInspectorProps {
  evidence: Evidence[];
  stats: PipelineStats;
  source: Source;
}

export const EvidenceInspector: React.FC<EvidenceInspectorProps> = ({
  evidence,
  stats,
  source,
}) => {
  const [filterText, setFilterText] = useState('');
  const [selectedTicker, setSelectedTicker] = useState<string>('ALL');

  const tickers = Array.from(new Set(evidence.map((e) => e.ticker)));

  const filtered = evidence.filter((item) => {
    const matchesTicker =
      selectedTicker === 'ALL' || item.ticker === selectedTicker;
    const q = filterText.trim().toLowerCase();
    const matchesText =
      !q ||
      item.ticker.toLowerCase().includes(q) ||
      item.claim_id.toLowerCase().includes(q) ||
      item.text.toLowerCase().includes(q) ||
      item.quote.toLowerCase().includes(q) ||
      item.accession.toLowerCase().includes(q);
    return matchesTicker && matchesText;
  });

  return (
    <div className="space-y-5">
      {/* Pipeline Extraction Telemetry Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div className="text-[11px] uppercase tracking-wider text-slate-400">
            Filings Fetched
          </div>
          <div className="mt-1 text-lg font-bold text-slate-100">
            <TaggedNumber value={stats.filings_fetched} source={source} />
          </div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div className="text-[11px] uppercase tracking-wider text-slate-400">
            Item 1A Parsed OK
          </div>
          <div className="mt-1 text-lg font-bold text-emerald-300">
            <TaggedNumber value={stats.parsed_ok} source={source} />
          </div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div className="text-[11px] uppercase tracking-wider text-slate-400">
            Verbatim Quote Match
          </div>
          <div className="mt-1 text-lg font-bold text-slate-100">
            <TaggedNumber
              value={`${(stats.quote_verify_rate * 100).toFixed(1)}%`}
              source={source}
            />
          </div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div className="text-[11px] uppercase tracking-wider text-slate-400">
            JSON Repair Rate
          </div>
          <div className="mt-1 text-lg font-bold text-slate-100">
            <TaggedNumber
              value={`${(stats.repair_rate * 100).toFixed(1)}%`}
              source={source}
            />
          </div>
        </div>
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div className="text-[11px] uppercase tracking-wider text-slate-400">
            Dropped Claims (Unverified)
          </div>
          <div
            className={`mt-1 text-lg font-bold ${
              stats.dropped_claims > 0 ? 'text-amber-300' : 'text-slate-100'
            }`}
          >
            <TaggedNumber value={stats.dropped_claims} source={source} />
          </div>
        </div>
      </div>

      {/* Evidence Inspector List */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <FileCheck2 className="w-4 h-4 text-emerald-400" />
              <h2 className="text-base font-semibold text-slate-100">
                SEC EDGAR Verbatim Evidence & Citation Inspector
              </h2>
              <SourceBadge source={source} size="sm" />
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Every extracted claim is verified as an exact substring of the
              source 10-K filing with timezone-aware SEC acceptance timestamps.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
              <input
                type="text"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                placeholder="Filter quote, claim, accession..."
                className="bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/60"
              />
            </div>

            <div className="inline-flex rounded-lg bg-slate-950 p-0.5 border border-slate-800">
              <button
                onClick={() => setSelectedTicker('ALL')}
                className={`px-2.5 py-1 rounded-md text-xs font-mono ${
                  selectedTicker === 'ALL'
                    ? 'bg-slate-800 text-slate-100 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                ALL ({evidence.length})
              </button>
              {tickers.map((t) => (
                <button
                  key={t}
                  onClick={() => setSelectedTicker(t)}
                  className={`px-2.5 py-1 rounded-md text-xs font-mono ${
                    selectedTicker === t
                      ? 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/30 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-4">
          {filtered.map((item) => (
            <div
              key={item.claim_id}
              className="bg-slate-950/70 border border-slate-800/90 rounded-xl p-4 hover:border-slate-700 transition-colors"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono text-xs font-bold">
                    {item.ticker}
                  </span>
                  <span className="font-mono text-xs text-slate-400">
                    {item.claim_id}
                  </span>
                  <SourceBadge source={source} size="xs" />
                </div>

                <div className="flex flex-wrap items-center gap-3 text-xs font-mono text-slate-400">
                  <span>
                    Accession:{' '}
                    <span className="text-slate-200">{item.accession}</span>
                  </span>
                  <span>
                    Accepted (TZ):{' '}
                    <span className="text-emerald-300">{item.accepted_at}</span>
                  </span>
                  <a
                    href={item.filing_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-slate-700 transition-colors"
                  >
                    <span>SEC EDGAR Filing</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>

              <p className="text-sm text-slate-200 font-medium mb-2.5">
                {item.text}
              </p>

              <blockquote className="border-l-2 border-emerald-500/60 bg-slate-900/60 pl-3.5 py-2 pr-3 rounded-r-lg text-xs font-mono text-slate-300 leading-relaxed">
                <span className="text-[10px] uppercase tracking-wider text-emerald-400 block mb-1">
                  Verbatim 10-K Substring Match:
                </span>
                “{item.quote}”
              </blockquote>
            </div>
          ))}

          {filtered.length === 0 && (
            <div className="text-center py-8 text-xs text-slate-400">
              No evidence citations match your current filter.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
