import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import fs from 'fs';
import path from 'path';
import App from '../App';

const replaysDir = path.resolve(__dirname, '../../public/replays');
const indexData = JSON.parse(
  fs.readFileSync(path.join(replaysDir, 'index.json'), 'utf-8')
);
const mockSampleData = JSON.parse(
  fs.readFileSync(path.join(replaysDir, 'mock-sample.json'), 'utf-8')
);
const realRunData = JSON.parse(
  fs.readFileSync(path.join(replaysDir, 'real-groq-run-01.json'), 'utf-8')
);

describe('Albus Frontend UI Contract & Source Tagging Verification', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/replays/index.json') || url.endsWith('/api/runs')) {
        return new Response(JSON.stringify(indexData), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.includes('mock-sample')) {
        return new Response(JSON.stringify(mockSampleData), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.includes('real-groq-run-01')) {
        return new Response(JSON.stringify(realRunData), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (url.endsWith('/api/config')) {
        return new Response(
          JSON.stringify({
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
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      return new Response('Not Found', { status: 404 });
    });
  });

  it('renders mock-sample.json with mandatory MOCK banner, caveat alerts, and MOCK source tags on numbers', async () => {
    render(<App />);

    // Wait for mock-sample (first in index.json) to render
    await waitFor(() => {
      expect(screen.getByTestId('mock-warning-banner')).toBeInTheDocument();
    });

    // Verify hypothesis starts with MOCK SAMPLE:
    expect(
      screen.getAllByText(/MOCK SAMPLE: Companies expanding cybersecurity/i)
        .length
    ).toBeGreaterThan(0);

    // Verify caveat state banner is shown because of the failed critic check
    expect(screen.getByTestId('caveat-banner')).toBeInTheDocument();
    expect(
      screen.getAllByText(/Citation Verbatim Substring/i).length
    ).toBeGreaterThan(0);

    // Verify MOCK source badges are attached to numbers
    const mockBadges = screen.getAllByTestId('source-badge-MOCK');
    expect(mockBadges.length).toBeGreaterThan(15);

    // Verify both calibration tabs exist and can be toggled without mixing
    const retroTab = screen.getByRole('tab', { name: /RETROSPECTIVE_CLEAN/i });
    const liveTab = screen.getByRole('tab', { name: /LIVE_RESOLVED/i });
    expect(retroTab).toBeInTheDocument();
    expect(liveTab).toBeInTheDocument();

    fireEvent.click(liveTab);
    await waitFor(() => {
      expect(screen.getByText(/Reliability Diagram \(live\)/i)).toBeInTheDocument();
    });
  });

  it('switches to real-groq-run-01.json and displays REAL verified banner and REAL source tags', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId('mock-warning-banner')).toBeInTheDocument();
    });

    // Select real-groq-run-01 from the dropdown
    const selector = screen.getByLabelText(/Select Replay Run/i);
    fireEvent.change(selector, { target: { value: 'real-groq-run-01' } });

    await waitFor(() => {
      expect(screen.getByTestId('real-mode-banner')).toBeInTheDocument();
    });

    expect(
      screen.getAllByText(
        /Year-over-year Item 1A character expansion correlates/i
      ).length
    ).toBeGreaterThan(0);

    const realBadges = screen.getAllByTestId('source-badge-REAL');
    expect(realBadges.length).toBeGreaterThan(15);
  });
});
