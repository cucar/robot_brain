# Implementation Plan: Repo Reorganization + Trading System

## Context

The machine_intelligence monorepo contains the Robot Brain prediction engine (Rust core + N-API bindings), Node.js wrapper libraries, and several demo apps including a stock trading simulation. The goal is to:

1. **Reorganize machine_intelligence** as a publishable open-source project — Rust crate on crates.io, platform-specific Node.js native binaries and a high-level wrapper on npm. All Node.js packages live under `libs/node/` as separate publishable packages.
2. **Create robot-brain-trader** as a separate private repo that depends on the published `robot-brain` npm package and implements the automated daily trading system (Express + Preact + MySQL, deployed on Digital Ocean with Nginx + pm2).

### Published Packages

| Package | Scope | Description |
|---------|-------|-------------|
| `robot-brain` (crate) | crates.io | Rust core prediction engine |
| `@robot-brain/core` | npm | Platform detection + native binary loader + TypeScript types |
| `@robot-brain/win32-x64` | npm | Prebuilt .node binary for Windows x64 |
| `@robot-brain/linux-x64` | npm | Prebuilt .node binary for Linux x64 |
| `robot-brain` | npm (unscoped) | Top-level wrapper — re-exports Brain from @robot-brain/core |
| `robot-brain-jobs` | npm (unscoped) | Job base class, runJob, parseBrainArgs, renderer |

---

## Part 1: machine_intelligence Reorganization

### New Directory Layout

```
machine_intelligence/
├── .github/workflows/
│   ├── ci.yml                        # PR checks: cargo test/clippy + cross-platform build
│   └── release.yml                   # Tag-triggered: build binaries, publish to npm + crates.io
├── brain/
│   ├── Cargo.toml                    # Workspace (unchanged)
│   ├── brain-core/
│   │   ├── Cargo.toml                # [package] name = "robot-brain" (for crates.io)
│   │   └── src/                      # All Rust source (unchanged)
│   ├── brain-napi/
│   │   ├── Cargo.toml                # (unchanged, internal only)
│   │   ├── build.rs
│   │   └── src/lib.rs
│   └── build.ps1                     # Local dev convenience (unchanged)
├── libs/node/
│   ├── core/                         # @robot-brain/core
│   │   ├── package.json              # optionalDeps: @robot-brain/win32-x64, linux-x64
│   │   ├── src/index.js              # Platform detection + native binary loading
│   │   └── index.d.ts               # TypeScript types (moved from brain-napi/)
│   ├── robot-brain/                  # robot-brain (unscoped wrapper)
│   │   ├── package.json              # deps: @robot-brain/core
│   │   └── index.js                  # Re-exports Brain from @robot-brain/core
│   ├── jobs/                         # robot-brain-jobs
│   │   ├── package.json              # deps: robot-brain
│   │   ├── index.js                  # Exports Job, runJob, executeJob, parseBrainArgs
│   │   └── src/
│   │       ├── job.js                # Job base class
│   │       ├── run.js                # CLI runner + arg parser
│   │       └── renderer.js           # Frame summary renderer
│   ├── win32-x64/                    # @robot-brain/win32-x64
│   │   ├── package.json              # os: ["win32"], cpu: ["x64"]
│   │   └── brain-napi.node           # Prebuilt binary (CI artifact, gitignored)
│   └── linux-x64/                    # @robot-brain/linux-x64
│       ├── package.json              # os: ["linux"], cpu: ["x64"]
│       └── brain-napi.node           # Prebuilt binary (CI artifact, gitignored)
├── apps/
│   ├── text/                         # Text demo (stays)
│   └── db/                           # MySQL brain backup utilities (stays)
├── docs/                             # Architecture docs (unchanged)
├── images/                           # README charts (unchanged)
├── pnpm-workspace.yaml               # packages: ['libs/node/*', 'apps/*']
├── .gitignore
├── README.md
└── LICENSE                           # Apache 2.0
```

### What Changes

| Current | New | Notes |
|---------|-----|-------|
| `brain/src/index.js` | `libs/node/core/src/index.js` | Rewritten: platform detection via `os.platform()`+`os.arch()`, requires `@robot-brain/<platform>`. Env var `ROBOT_BRAIN_NATIVE_PATH` for local dev override. |
| `brain/brain-napi/index.d.ts` | `libs/node/core/index.d.ts` | Moved as-is |
| `libs/node/index.js` | Split into two packages | Brain re-export → `libs/node/robot-brain/index.js`. Job/run/renderer → `libs/node/jobs/index.js` |
| `libs/node/src/job.js` | `libs/node/jobs/src/job.js` | Import changed from `'brain'` to `'robot-brain'` |
| `libs/node/src/run.js` | `libs/node/jobs/src/run.js` | Import changed similarly |
| `libs/node/src/renderer.js` | `libs/node/jobs/src/renderer.js` | No changes |
| `libs/node/package.json` | Deleted | Replaced by `libs/node/robot-brain/` and `libs/node/jobs/` |
| `brain/brain-core/Cargo.toml` | Same file | `[package] name` changed to `"robot-brain"`, add crates.io metadata |
| `pnpm-workspace.yaml` | Same file | Updated from `['brain', 'libs/*', 'apps/*']` to `['libs/node/*', 'apps/*']` |
| `brain/package.json` | Deleted | Replaced by `libs/node/core/` |
| `apps/stocks/` | Stays as-is | Proof-of-concept demo; code is adapted (not moved) to robot-brain-trader |

### Package Dependency Chain

```
@robot-brain/win32-x64  ─┐
@robot-brain/linux-x64  ─┤ (optionalDeps)
                          ▼
                  @robot-brain/core
                          │ (dep)
                          ▼
                     robot-brain
                          │ (dep)
                          ▼
                   robot-brain-jobs
```

### Platform Binary Pattern (esbuild/swc-style)

- Each `@robot-brain/<platform>` package has `os` and `cpu` fields in package.json so npm/pnpm only installs the matching one
- `@robot-brain/core` lists all platform packages as `optionalDependencies`
- At runtime, `libs/node/core/src/index.js` builds the package name from `os.platform()` + `os.arch()` and calls `createRequire(import.meta.url)('@robot-brain/<platform>')`
- CI builds the .node binary on each OS, copies it into the corresponding `libs/node/<platform>/` dir, and publishes
- For local dev, env var `ROBOT_BRAIN_NATIVE_PATH` overrides the package lookup to load a local build artifact directly

### GitHub Actions CI

**ci.yml** (on PR):
- `rust-check` job: cargo check + test + clippy (ubuntu-latest)
- `build-native` job (matrix: windows-latest, ubuntu-latest): cargo build --release -p brain-napi, upload .node artifact
- `node-test` job (needs build-native): download artifact, pnpm install, verify loading works

**release.yml** (on tag `v*`):
- Build native binaries on both platforms
- Publish in dependency order: win32-x64, linux-x64 → core → robot-brain → robot-brain-jobs
- Publish robot-brain crate to crates.io

---

## Part 2: robot-brain-trader (New Private Repo)

### Tech Stack
- **Backend**: Express + mysql2/promise (no ORM)
- **Frontend**: Preact + Vite + uPlot (charts) + preact-router
- **Database**: MySQL
- **Scheduling**: system crontab (Fedora)
- **Brokerage**: @alpacahq/alpaca-trade-api (paid plan)
- **Deployment**: Digital Ocean (Fedora 38), Nginx reverse proxy, pm2 process manager

### Directory Structure

```
robot-brain-trader/
├── .env.example
├── package.json                      # private, scripts for dev/build/start
├── ecosystem.config.cjs              # pm2 config
├── nginx.conf.example                # Nginx reverse proxy config template
├── src/
│   ├── shared/                           # Used by both server and job
│   │   ├── alpaca.js                     # Alpaca API client (orders, positions, account, bars)
│   │   ├── notifier.js                   # Webhook/email notifications
│   │   ├── db.js                         # MySQL connection pool (mysql2/promise)
│   │   └── config.js                     # Env var parsing + validation
│   ├── server/
│   │   ├── index.js                      # Express bootstrap, mount routes, serve SPA
│   │   ├── middleware/
│   │   │   └── auth.js                   # Access-key guard
│   │   └── routes/
│   │       ├── api-portfolio.js          # GET /api/portfolio, /api/positions
│   │       ├── api-trades.js             # GET /api/trades, /api/trades/:id
│   │       ├── api-signals.js            # GET /api/signals, /api/signals/:symbol
│   │       ├── api-performance.js
│   │       ├── api-controls.js           # POST pause/resume/emergency-stop/liquidate
│   │       ├── api-config.js             # GET/PUT /api/config
│   │       └── api-status.js             # GET /api/status
│   ├── job/                              # Standalone crontab script
│   │   ├── trading-job.js                # Entry point (crontab runs this directly)
│   │   ├── encoder.js                    # StockEncoder (adapted from apps/stocks/)
│   │   ├── portfolio.js                  # Allocation, sizing, diff logic
│   │   └── steps/
│   │       ├── snapshot.js               # 1. Fetch quotes for 100 symbols
│   │       ├── predict.js                # 2. Feed brain, collect signals
│   │       ├── select.js                 # 3. Rank + select top N
│   │       ├── diff.js                   # 4. Diff desired vs actual positions
│   │       ├── execute-sells.js          # 5. Limit sells at bid, poll 5 min
│   │       ├── execute-buys.js           # 6. Limit buys at ask, poll 5 min
│   │       └── report.js                 # 7. Log + notify
│   ├── admin/                            # Preact SPA — administration panel
│   │   ├── index.html                    # Vite entry
│   │   ├── index.jsx                     # Preact app root + router
│   │   ├── api.js                        # Fetch wrapper for /api/*
│   │   ├── components/
│   │   │   ├── Layout.jsx                # Nav sidebar + content area
│   │   │   └── shared/                   # Table, Chart, StatusBadge, ConfirmDialog
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx             # P&L, slippage summary, value chart, drill-down to trades
│   │   │   ├── TradeLog.jsx              # Filterable by date range, symbol, side; paginated
│   │   │   ├── Signals.jsx               # 100-stock signal table
│   │   │   ├── Performance.jsx           # Equity curve, Sharpe, drawdown
│   │   │   ├── Controls.jsx              # Kill switch toggle, liquidate all, job status indicator
│   │   │   ├── Logs.jsx                  # View daily job output logs (date picker)
│   │   │   ├── Configuration.jsx         # Watchlist, sizing, notifications
│   │   │   └── Login.jsx
│   │   └── hooks/                        # useApi, useAuth
├── data/historical/1D/               # CSV training data
├── migrations/
│   ├── 001-initial-schema.sql
│   └── run.js                        # Migration runner
├── scripts/
│   ├── download-data.js              # Fetch historical bars from Alpaca API
│   ├── process-data.js               # Clean raw Alpaca output → encoder-ready CSVs
│   └── train.js                      # Train brain on historical data (2 episodes), save state
├── logs/                             # Job output logs (one file per day, rotated 30 days)
├── vite.config.js
└── README.md
```

### Deployment Architecture (Digital Ocean + Nginx + pm2)

- **Domain**: `robot-brain.org` (registered via GoDaddy)
- **DNS**: A record pointing `robot-brain.org` to the Digital Ocean droplet's IP. Optional: `www` CNAME → `robot-brain.org`
- **Nginx** serves as reverse proxy: routes `/api/*` and `/admin/*` to the Express app running on localhost:3000
- **SSL**: Let's Encrypt via certbot with auto-renewal (`certbot --nginx -d robot-brain.org -d www.robot-brain.org`)
- **pm2** manages the Express process (auto-restart on crash, log rotation)
- `ecosystem.config.cjs` defines the pm2 app config (name, script, env vars, instances)
- `nginx.conf.example` provides the server block template with SSL and domain config
- Static admin build (`dist/`) served by Express directly (or optionally by Nginx for better perf)
- Brain state stored on disk under `BRAIN_BACKUP_DIR` (persistent across restarts)

### Domain Setup (Phase 9)

1. **GoDaddy**: Purchase `robot-brain.org`, go to DNS Management
2. **DNS Records**:
   - `A` record: `@` → `<droplet-ip>`
   - `CNAME` record: `www` → `robot-brain.org`
3. **Nginx server block** (`/etc/nginx/conf.d/robot-brain.conf`):
   - `server_name robot-brain.org www.robot-brain.org;`
   - `proxy_pass http://127.0.0.1:3000;`
4. **SSL**: `sudo certbot --nginx -d robot-brain.org -d www.robot-brain.org`
5. **Verify**: `https://robot-brain.org` loads the admin login page

### MySQL Schema (7 tables)

- **trading_day** — date, status (pending/running/completed/failed/skipped), portfolio_value, cash_balance, net_pnl, timestamps
- **signal** — trading_day_id FK, symbol, action (OWN/OUT), strength, rank, selected boolean
- **order** — trading_day_id FK, alpaca_order_id, symbol, side, quantity, snapshot_price, limit_price, filled_quantity, filled_avg_price, slippage_bps, status, timestamps
- **position** — symbol (unique), shares, avg_cost, current_price, market_value, unrealized_pnl, opened_at
- **portfolio_snapshot** — trading_day_id FK, snapshot_type (open/post_sell/post_buy/close), total_value, cash, positions_value
- **config** — key-value store (watchlist, max_positions, paused, etc.)
- **alert_log** — level, category, message, metadata JSON, acknowledged boolean

### API Routes (20 endpoints)

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/auth/login | Validate access key |
| GET | /api/status | System health + job status |
| GET | /api/portfolio | Portfolio snapshot: value, P&L, spread/slippage summary |
| GET | /api/positions | Open positions with unrealized P&L |
| GET | /api/trades | Paginated trade log (filter by start/end date, symbol, side, status) |
| GET | /api/trades/:id | Single trade detail with snapshot/limit/fill prices |
| GET | /api/signals | Latest signals (query: ?date=) |
| GET | /api/signals/:symbol | Signal history for one symbol |
| GET | /api/performance | Equity curve, Sharpe, drawdown, per-stock stats |
| POST | /api/controls/kill | Activate kill switch (stops all future job runs) |
| POST | /api/controls/unkill | Deactivate kill switch (resume job runs) |
| POST | /api/controls/liquidate | Liquidate all positions + activate kill switch (requires confirmation token) |
| GET | /api/config | Read config |
| PUT | /api/config | Update config |
| POST | /api/job/trigger | Manually trigger daily job |
| GET | /api/job/status | Is the job currently running? (checks OS process) |
| GET | /api/logs | List available log dates |
| GET | /api/logs/:date | Get log file content for a specific date |
| GET | /api/slippage | Aggregate spread/slippage stats since inception |
| GET | /api/slippage/:symbol | Per-symbol slippage breakdown |

### Daily Job Pipeline

Sequential pipeline, stateless on restart (reads all state from Alpaca). All output logged to `logs/YYYY-MM-DD.log` (one file per day, rotated after 30 days).

0. **Kill Switch Check** — Read `config.kill_switch` from DB. If true, log "kill switch active", exit immediately. Also check if market is open today (Alpaca calendar API).
1. **Snapshot** — getBars for 100 symbols, record opening portfolio_snapshot
2. **Predict** — Load brain from disk, resetContext, encode all symbols, processFrame, extract OWN/OUT + strength per symbol, save brain
3. **Select** — Rank OWN signals by strength, filter by max price, take top N
4. **Diff** — Read positions from Alpaca, compare desired vs actual, produce sell/buy lists
5. **Execute Sells** — Limit sell at bid, poll 10s intervals, 5-min timeout, cancel unfilled. Record snapshot price vs limit price placed for slippage tracking.
6. **Check Buying Power** — Scale down buys if insufficient (from Alpaca account query)
7. **Execute Buys** — Limit buy at ask, poll 10s intervals, 5-min timeout, cancel unfilled. Record snapshot price vs limit price placed for slippage tracking.
8. **Report** — Update trading_day, record close snapshot, send notification

Brain state persists via file-based backup (existing Rust backup module). Context resets daily. The trading_day.status column prevents double-execution.

### Kill Switch & Liquidation

Two emergency controls accessible from the admin panel:

- **Kill Switch** (`config.kill_switch` in DB) — When activated, the daily job reads this flag before doing anything and exits immediately. No trades placed. The admin shows a prominent red indicator when active. Toggle on/off from Controls page.
- **Liquidate All** — When pressed (with double-confirmation), the server immediately: cancels all open Alpaca orders, places market sell orders for all positions, sets `config.kill_switch = true` to prevent the next job from trading. Goes to 100% cash.

### Job Logging

- The job writes all output to `logs/YYYY-MM-DD.log` (stdout + stderr tee'd to file)
- One log file per calendar day, rotated after 30 days (older files deleted on job start)
- The admin panel can view logs via `GET /api/logs` (lists available dates) and `GET /api/logs/:date` (returns log content)
- Current job run status visible in admin via `GET /api/job/status` — checks if the process is running (`ps aux | grep trading-job`)

### Spread & Slippage Tracking

Every order records three prices in the `order` table:
- **snapshot_price** — The price at the moment the brain made its prediction
- **limit_price** — The bid (sell) or ask (buy) at the moment the order was placed
- **fill_price** — The actual execution price from Alpaca

This allows computing:
- **Prediction-to-fill slippage** = fill_price - snapshot_price (what the brain expected vs. what happened)
- **Order slippage** = fill_price - limit_price (how much the market moved between order placement and fill)
- Aggregated across all trades since inception, displayed on the Dashboard

### Code Migration from machine_intelligence

| Source | Destination | Change Level |
|--------|-------------|-------------|
| `apps/stocks/encoder.js` | `src/job/encoder.js` | Minor (remove simulation-only statics) |
| `apps/stocks/trader.js` | `src/job/portfolio.js` | **Major rewrite** — DB-backed positions, Alpaca orders replace in-memory simulation. Allocation ranking logic preserved. |
| `apps/stocks/alpaca.js` | `src/shared/alpaca.js` | Extend with order submission, position query, account info, cancel. Remove rate limiting. |
| `apps/stocks/jobs/download.js` | `scripts/download-data.js` | Path adjustments |
| `apps/stocks/jobs/setup.js` | `scripts/process-data.js` | Path adjustments |
| `apps/stocks/data/1D/*.csv` | `data/historical/1D/` | Copy as-is |
| `apps/db/database.js` | `src/shared/db.js` | Refactor to connection pool |

---

## Implementation Phases

### Phase 0: Parameter Optimization — 1D (prerequisite, in machine_intelligence repo)

Before building the trading system, find the optimal brain parameters for 1D stock data. All tests use the existing `apps/stocks/` infrastructure.

**Test methodology:**
- 5 years of data per test: pre-train on 4 years (2 episodes) + test on the final 1 year
- Change one parameter at a time, hold others at known safe/max values
- Use 100 threads (columns) for parallelism
- Chart the effect of each parameter independently

**Parameters to optimize:**
1. Portfolio stock count — how many positions to hold simultaneously
2. Trained stock count — how many symbols to train on: 3 (30 samples), 10 (10 samples), 30 (5 samples), 50 (5 samples), 100 (1 sample)
3. Error correction threshold — aggressive, mean, loose, or static
4. Merge threshold
5. Context length
6. Pattern forget rate
7. Max portfolio stock price

**Crisis validation:**
- Train on 2000–2007 data (2 episodes), test on 2008–2009
- Verify the model doesn't blow up during a crash — does it go to cash? Does it recover?
- This is a go/no-go gate for live trading

**Output:** Optimal parameter set for 1D trading, documented with charts. Brain config locked for production.

**Verify:** Each parameter has a chart showing its effect on returns/Sharpe. Optimal parameter set documented. Crisis test (2000–2007 train → 2008–2009 test) passes go/no-go criteria.

---

### Phase 1: Restructure machine_intelligence (1-2 sessions)
1. Create `libs/node/` subpackage directories: `core/`, `robot-brain/`, `jobs/`, `win32-x64/`, `linux-x64/`
2. Move/rewrite `brain/src/index.js` → `libs/node/core/src/index.js` (platform detection)
3. Move `brain/brain-napi/index.d.ts` → `libs/node/core/index.d.ts`
4. Split current `libs/node/index.js`: Brain re-export → `libs/node/robot-brain/index.js`, Job/run/renderer → `libs/node/jobs/`
5. Move `libs/node/src/job.js`, `run.js`, `renderer.js` → `libs/node/jobs/src/`
6. Create all package.json files for the 5 npm packages
7. Update `pnpm-workspace.yaml` to `['libs/node/*', 'apps/*']`
8. Update `apps/text/package.json` to depend on `robot-brain-jobs` (workspace ref)
9. Delete old `libs/node/package.json`, `libs/node/index.js`, `libs/node/src/`, `brain/package.json`
10. Rename brain-core crate to `robot-brain` in `brain/brain-core/Cargo.toml`, add crates.io metadata
11. Update `.gitignore` to include `libs/node/*/brain-napi.node`

**Documentation (this phase):**
- Rust crate `robot-brain`: `//!` crate-level doc in `lib.rs`, doc comments on all `pub` methods in `brain.rs`, `types.rs`, and public items in internal modules. `brain/brain-core/README.md` for crates.io.
- `@robot-brain/core`: `README.md` — platform binary loader explanation, supported platforms, usage
- `robot-brain`: `README.md` — primary user-facing docs: install, quick-start, API reference, platform support
- `robot-brain-jobs`: `README.md` — Job subclass pattern, `runJob()`/`parseBrainArgs()`, CLI flags, example
- Platform packages: short `README.md` — not meant for direct install, link to `robot-brain`
- Top-level `README.md`: add "Packages" section with npm/crates.io badges, update install instructions

**Verify:** Run `brain/build.ps1`, copy .node to `libs/node/win32-x64/`, run `node apps/text/jobs/test.js` and `node apps/stocks/jobs/test.js` — both load Brain via new platform detection path.

### Phase 2: GitHub Actions CI/CD (1 session)
1. Create `.github/workflows/ci.yml` (rust-check + build-native matrix + node-test)
2. Create `.github/workflows/release.yml` (tag-triggered publish in dependency order)
3. Create `scripts/bump-version.js` for lockstep version bumping across all package.json files
4. Set up npm org `@robot-brain`, npm + crates.io tokens as GitHub secrets
5. Test with a `v0.1.0` tag push

### Phase 3: Initialize robot-brain-trader (1 session)
1. Create private repo, scaffold directory structure
2. Install deps: express, mysql2, dotenv, robot-brain, @alpacahq/alpaca-trade-api
3. Create `migrations/001-initial-schema.sql` with full schema
4. Create `migrations/run.js` migration runner
5. Create `src/shared/config.js`, `src/shared/db.js`, `src/server/middleware/auth.js`, Express skeleton
6. Create `ecosystem.config.cjs` (pm2) and `nginx.conf.example`

**Verify:** Run migrations against local MySQL, hit health endpoint.

### Phase 4: Core Services (2-3 sessions)
1. Build `src/shared/alpaca.js` — extend with order submission, position query, account info, cancel (no rate limiting)
2. Build `src/shared/notifier.js` — webhook notifications
3. Adapt `src/job/encoder.js` — remove simulation-only statics
4. Build `src/job/portfolio.js` — allocation/sizing/diff logic (major rewrite from trader.js statics → Alpaca-backed)
5. Copy historical data to `data/historical/1D/`, adapt `scripts/download-data.js` and `scripts/process-data.js`

### Phase 5: Trading Job Pipeline (2-3 sessions)
1. Build all 7 step modules under `src/job/steps/`
2. Build `trading-job.js` orchestrator — kill switch check, market day check, sequential pipeline, error handling
3. Add file logging: tee stdout/stderr to `logs/YYYY-MM-DD.log`, rotate files older than 30 days on startup
4. Record snapshot_price, limit_price, fill_price on every order for slippage tracking

**Verify:** Configure Alpaca paper trading credentials. Manually run `node src/job/trading-job.js`. Confirm: kill switch check logged, signals written to DB, orders have snapshot/limit/fill prices, log file created in `logs/`, positions match in Alpaca.

### Phase 6: API Routes (1-2 sessions)
1. Implement all 20 route handlers (portfolio, trades, signals, performance, controls, config, job status, logs, slippage)
2. Kill switch endpoints: POST /api/controls/kill, /api/controls/unkill, /api/controls/liquidate
3. Log endpoints: GET /api/logs (list dates), GET /api/logs/:date (file content)
4. Job status endpoint: GET /api/job/status (check running process via `ps`)
5. Slippage endpoints: GET /api/slippage, GET /api/slippage/:symbol

**Verify:** Test each endpoint against populated DB + live Alpaca paper account.

### Phase 7: Preact Admin Panel (3-4 sessions)
1. Set up Vite + Preact + preact-router (`src/admin/`)
2. Build `Layout.jsx` (nav sidebar + content), `Login.jsx`, auth hooks
3. **Dashboard** — Total P&L since inception, aggregate slippage stats, historical portfolio value chart (uPlot). Clicking P&L/slippage drills down to TradeLog with filters pre-applied.
4. **TradeLog** — Paginated table with date range filter (start/end), symbol, side. Each row shows snapshot price, limit price, fill price, slippage. Click for full detail.
5. **Controls** — Kill switch toggle (red on/off with confirmation), Liquidate All button (double-confirmation), current job status indicator (running/idle/killed via `ps` check)
6. **Logs** — Date picker, displays log file content for selected day
7. **Signals, Performance, Configuration** — As described in the API routes section
8. Build shared components: Table, Chart (uPlot wrapper), StatusBadge, ConfirmDialog
9. Wire to API, test in browser against running server

**Verify:** Start dev server (Vite + Express). Log in with access key. Dashboard: P&L and slippage stats load, value chart renders, clicking drills down to trade log. TradeLog: date range filter works, pagination works, shows snapshot/limit/fill prices. Controls: kill switch toggles on/off, liquidate shows double-confirmation, job status updates. Logs: date picker lists dates, shows content. Test kill switch end-to-end: activate → manually trigger job → confirm it exits immediately.

### Phase 8: Training & Crisis Validation (1-2 sessions)
1. Build `scripts/train.js` — Runs brain through historical data for N episodes, saves trained state
2. Download latest historical data via `scripts/download-data.js`, clean with `scripts/process-data.js`
3. **Crisis test**: Train on 2000–2007 (2 episodes), run against 2008–2009. Analyze: does the model go to cash during the crash? How deep is the drawdown? Does it recover?
4. **Production training**: Train on the longest available data window (2 episodes) using the optimized parameters from Phase 0
5. Save the final trained brain state — this is the starting point for live trading

**Verify:** Crisis test results reviewed and documented. Trained brain state file loads successfully. Brain produces sensible OWN/OUT signals on recent data (sanity check, not a backtest).

### Phase 9: Deployment (1 session)
1. Register `robot-brain.org` on GoDaddy
2. Set DNS A record: `@` → droplet IP, CNAME: `www` → `robot-brain.org`
3. Set up MySQL on Digital Ocean server
4. Clone repo, install deps, run migrations
5. Build admin panel (`vite build`)
6. Configure `ecosystem.config.cjs`, start with `pm2 start`
7. Configure Nginx server block for `robot-brain.org`, enable reverse proxy to :3000
8. Run `certbot --nginx -d robot-brain.org -d www.robot-brain.org` for SSL
9. Upload trained brain state to server
10. Set environment variables, configure crontab entry

**Verify:** `pm2 status` shows app running. `curl localhost:3000/api/status` returns JSON. `https://robot-brain.org` loads admin login. SSL certificate valid (`certbot certificates`). Cron fires at 10:15 AM ET on next market day. Notification webhook fires after job completes. Kill switch works end-to-end from admin panel.

**Total: ~14-22 sessions**

---

## Environment Variables (.env.example)

```
# Alpaca
ALPACA_KEY_ID=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true

# MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=
DB_PASSWORD=
DB_NAME=robot_brain_trader

# Server
PORT=3000
```

Brain parameters (backup dir, context length, regions, columns) are passed as command-line arguments to the cron job, not as server environment variables. The crontab entry handles this:
```
15 10 * * 1-5  cd /path/to/robot-brain-trader && node src/job/trading-job.js --brain-dir ./brain-state --context-length 10 --regions 1 --columns 1
```
