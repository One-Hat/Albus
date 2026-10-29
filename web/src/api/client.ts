import type {
  RunIndexItem,
  RunSummary,
  StageEvent,
} from '@contracts/types';

export interface ApiConfig {
  provider: string;
  models: { small: string; mid: string; large: string };
  live_enabled: boolean;
  limits: {
    max_tickers: number;
    max_tokens_per_call: number;
    max_wall_time_sec: number;
    global_daily_budget_usd: number;
    rate_limit_rpm: number;
    concurrency_limit_per_ip: number;
  };
  backend_reachable: boolean;
}

export interface LedgerVerifyResult {
  verified: boolean;
  broken_at: number | null;
  length: number;
  head_hash: string;
  computed_hash?: string;
}

const API_BASE = (import.meta.env?.VITE_API_BASE_URL || import.meta.env?.NEXT_PUBLIC_API_BASE_URL || '').replace(/\/$/, '');

const FALLBACK_CONFIG: ApiConfig = {
  provider: 'groq',
  models: {
    small: 'openai/gpt-oss-20b',
    mid: 'openai/gpt-oss-120b',
    large: 'openai/gpt-oss-120b',
  },
  live_enabled: false,
  limits: {
    max_tickers: 20,
    max_tokens_per_call: 4096,
    max_wall_time_sec: 120,
    global_daily_budget_usd: 10.0,
    rate_limit_rpm: 20,
    concurrency_limit_per_ip: 1,
  },
  backend_reachable: false,
};

export async function fetchApiConfig(): Promise<ApiConfig> {
  try {
    const res = await fetch(`${API_BASE}/api/config`, {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      return FALLBACK_CONFIG;
    }
    const data = await res.json();
    if (!data || !data.models) {
      return FALLBACK_CONFIG;
    }
    return {
      ...data,
      backend_reachable: true,
    };
  } catch {
    return FALLBACK_CONFIG;
  }
}

export async function fetchRunIndex(): Promise<RunIndexItem[]> {
  try {
    const res = await fetch('/replays/index.json', {
      headers: { Accept: 'application/json' },
    });
    if (res.ok) {
      const items: RunIndexItem[] = await res.json();
      if (Array.isArray(items) && items.length > 0) {
        return items;
      }
    }
  } catch {
    // Fall back to /api/runs if static index fetch fails
  }

  const apiRes = await fetch(`${API_BASE}/api/runs`, {
    headers: { Accept: 'application/json' },
  });
  if (!apiRes.ok) {
    throw new Error(`Failed to load run index (HTTP ${apiRes.status})`);
  }
  return apiRes.json();
}

export async function fetchRunSummary(runId: string): Promise<RunSummary> {
  const safeId = encodeURIComponent(runId);
  try {
    const res = await fetch(`/replays/${safeId}.json`, {
      headers: { Accept: 'application/json' },
    });
    if (res.ok) {
      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/json') || contentType === '') {
        return await res.json();
      }
    }
  } catch {
    // Fall back to live API endpoint
  }

  const apiRes = await fetch(`${API_BASE}/api/runs/${safeId}`, {
    headers: { Accept: 'application/json' },
  });
  if (!apiRes.ok) {
    throw new Error(`Run '${runId}' could not be loaded (HTTP ${apiRes.status})`);
  }
  return apiRes.json();
}

async function sha256Hex(input: string): Promise<string> {
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const encoded = new TextEncoder().encode(input);
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoded);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
  return '';
}

export async function verifyRunLedger(
  runId: string,
  summary: RunSummary
): Promise<LedgerVerifyResult> {
  const safeId = encodeURIComponent(runId);
  try {
    const res = await fetch(`${API_BASE}/api/ledger/verify/${safeId}`, {
      headers: { Accept: 'application/json' },
    });
    if (res.ok) {
      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return await res.json();
      }
    }
  } catch {
    // Perform deterministic client-side WebCrypto SHA-256 verification
  }

  let prev = 'GENESIS';
  for (const ev of summary.evidence) {
    const block = `claim:${ev.claim_id}:ticker:${ev.ticker}`;
    const digest = await sha256Hex(`${prev}:${block}`);
    if (digest) {
      prev = digest;
    }
  }

  const verified =
    prev === summary.ledger.head_hash ||
    (summary.ledger.verified && summary.evidence.length === 0);

  return {
    verified,
    broken_at: verified ? null : 0,
    length: summary.ledger.length,
    head_hash: summary.ledger.head_hash,
    computed_hash: prev,
  };
}

export async function submitLiveRun(
  hypothesis: string,
  universe: string[]
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ hypothesis, universe }),
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) detail = errJson.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

export function replayStageEvents(
  summary: RunSummary,
  onEvent: (ev: StageEvent) => void,
  onComplete: () => void
): () => void {
  const totalTickers = summary.universe.length;
  const sequence: StageEvent[] = [
    {
      stage: 'ingest',
      status: 'start',
      done: 0,
      total: totalTickers,
      message: `Fetching SEC EDGAR 10-K filings for ${totalTickers} universe entries...`,
    },
    {
      stage: 'ingest',
      status: 'done',
      done: summary.stats.filings_fetched,
      total: summary.stats.filings_fetched,
      message: `Fetched ${summary.stats.filings_fetched} 10-K filings with tz-aware acceptance timestamps.`,
    },
    {
      stage: 'extract',
      status: 'progress',
      done: summary.stats.parsed_ok,
      total: summary.stats.filings_fetched,
      message: `Extracting Item 1A Risk Factors via ${summary.models.small} (${(summary.stats.quote_verify_rate * 100).toFixed(1)}% verbatim quote match)...`,
    },
    {
      stage: 'extract',
      status: 'done',
      done: summary.stats.parsed_ok,
      total: summary.stats.filings_fetched,
      message: `Parsed ${summary.stats.parsed_ok}/${summary.stats.filings_fetched} filings (${summary.stats.dropped_claims} unverifiable claims dropped).`,
    },
    {
      stage: 'research',
      status: 'done',
      message: `Synthesized semantic risk diffs & executed ${summary.backtest.n_events}-event next-open backtest via ${summary.models.mid}.`,
    },
    {
      stage: 'swarm',
      status: 'done',
      done: summary.swarm.n_forecasts,
      total: summary.swarm.n_forecasts,
      message: `Aggregated ${summary.swarm.n_forecasts} calibrated forecasts across ${summary.swarm.n_agents} swarm personas.`,
    },
    {
      stage: 'critic',
      status: 'done',
      message: `Adjudicated ${summary.verdict.checks.length} audit gates via ${summary.models.large} -> Verdict: ${summary.verdict.label.toUpperCase()}.`,
    },
    {
      stage: 'report',
      status: 'done',
      message: `Sealed SHA-256 ledger (${summary.ledger.length} blocks, head=${summary.ledger.head_hash.slice(0, 12)}...).`,
    },
  ];

  let idx = 0;
  const timer = window.setInterval(() => {
    if (idx < sequence.length) {
      onEvent(sequence[idx]);
      idx += 1;
    } else {
      window.clearInterval(timer);
      onComplete();
    }
  }, 280);

  return () => window.clearInterval(timer);
}
