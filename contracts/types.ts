// contract_version: '1.0'
export type Source = 'REAL' | 'MOCK';
export type VerdictLabel = 'pass' | 'weak' | 'reject';

export interface RunMeta {
  id: string; created_at: string; source: Source; contract_version: string;
  provider: string; models: { small: string; mid: string; large: string };
  hypothesis: string; universe: string[];
}
export interface Check { name: string; passed: boolean; evidence: string }
export interface Verdict { label: VerdictLabel; checks: Check[]; narrative: string }
export interface Backtest {
  n_events: number; n_dropped: number; dropped_reasons: Record<string, number>;
  k_variants: number; sharpe: number; sharpe_ci: [number, number];
  max_drawdown: number; total_return: number; turnover: number; cost_drag: number;
  equity_curve: { date: string; value: number }[];
}
export interface Evidence {
  claim_id: string; ticker: string; text: string; quote: string;
  accession: string; filing_url: string; accepted_at: string;
}
export interface Swarm {
  n_agents: number; n_forecasts: number;
  weights_over_time: { round: number; weights: Record<string, number> }[];
}
export interface Calibration {   // one entry per kind. Never mixed.
  kind: 'retrospective_clean' | 'live'; n_resolved: number;
  brier: Record<string, number>;              // ensemble, base_rate, single_large, single_mid, equal_weight_swarm
  brier_ci: Record<string, [number, number]>; ece: number;
  reliability: { lo: number; hi: number; mean_pred: number; freq: number; n: number }[];
}
export interface Ledger { length: number; head_hash: string; verified: boolean }
export interface PipelineStats {
  filings_fetched: number; parsed_ok: number; quote_verify_rate: number;
  repair_rate: number; dropped_claims: number;
}
export interface RunSummary extends RunMeta {
  verdict: Verdict; backtest: Backtest; evidence: Evidence[];
  swarm: Swarm; calibration: Calibration[]; ledger: Ledger; stats: PipelineStats;
}
export interface RunIndexItem {
  id: string; created_at: string; source: Source; hypothesis: string; verdict: VerdictLabel;
}
export interface StageEvent {
  stage: 'ingest' | 'extract' | 'research' | 'swarm' | 'critic' | 'report';
  status: 'start' | 'progress' | 'done' | 'error';
  done?: number; total?: number; message?: string;
}
