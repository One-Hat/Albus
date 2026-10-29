# ALBUS Web Frontend (`web/`)

React + TypeScript + Tailwind CSS frontend consuming `contracts/types.ts` (v1.0), static replays (`web/public/replays/`), and the FastAPI live/replay endpoints (`/api/*`).

## Ground Rules Enforced
- **Zero LLM Keys**: `GROQ_API_KEY` never enters `web/`.
- **Strict Contract**: Imports types directly from `../contracts/types.ts`.
- **Mandatory Source Tagging**: Every displayed number carries a `[REAL]` or `[MOCK]` badge (`<TaggedNumber>`). Synthetic runs display a sticky `MOCK DATA DEMO` warning banner.

## Commands

```bash
cd web
npm install
npm test          # Run Vitest unit & contract verification tests
npm run build     # TypeScript check + production bundle build (dist/)
npm run dev       # Start local dev server on http://localhost:3000
```
