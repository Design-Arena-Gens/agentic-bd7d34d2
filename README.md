# Gemini Forex Autotrader

End-to-end automated forex trading stack that couples a web control surface (Next.js) with a MetaTrader 5 execution bridge powered by Gemini AI. The bot monitors market structure, generates trade plans, sizes positions automatically, and sends orders to MT5 while reporting telemetry back to the dashboard.

## Architecture

```
Next.js Control Plane (Vercel ready)
│  ├─ Realtime status dashboard (app/page.tsx)
│  ├─ REST API for config + event ingestion (/api/*)
│  └─ Gemini analysis trigger (optional manual cycle)
│
└── Python Execution Bridge (python-bot)
    ├─ FastAPI service exposing /market and /trades
    ├─ Gemini-driven strategy engine
    ├─ MetaTrader5 order management + risk controls
    └─ Event relay back to Next.js `/api/events`
```

## Prerequisites

- Node.js 18+
- Python 3.10+ (Windows recommended for MT5)
- MetaTrader 5 terminal installed (`terminal64.exe`)
- Google Gemini API key with `gemini-1.5-pro` access

## Setup

### 1. Install Node dependencies

```bash
npm install
```

### 2. Configure environment variables

Create `.env.local` in the project root:

```bash
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-1.5-pro
EXECUTION_BRIDGE_URL=http://localhost:8770
```

Configure the Python bridge by copying the template:

```bash
cd python-bot
cp .env.example .env
```

Update `.env` with your MT5 credentials, Gemini key, and terminal path.

### 3. Install Python dependencies

```bash
pip install -r python-bot/requirements.txt
```

### 4. Launch the MT5 execution bridge (Windows host)

```bash
cd python-bot
python main.py
```

This will:

1. Launch MT5 from `MT5_TERMINAL_PATH`
2. Connect using `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER`
3. Start a FastAPI service at `http://localhost:8770`
4. Begin an autonomous analysis loop calling Gemini and placing trades

### 5. Run the control plane

```bash
npm run dev
```

Open http://localhost:3000 to monitor status, update risk settings, trigger manual cycles, and view logs.

## Deployment

1. Build the Next.js app
   ```bash
   npm run build
   ```
2. Deploy to Vercel
   ```bash
   vercel deploy --prod --yes --token $VERCEL_TOKEN --name agentic-bd7d34d2
   ```
3. Update `CONTROL_PLANE_URL` in the Python `.env` to point at the production URL (e.g. `https://agentic-bd7d34d2.vercel.app`) and restart `python main.py`.

> The Python bridge **must** remain on a Windows machine with MT5 installed. Vercel hosts the dashboard only; live trading happens from your execution host.

## Key Features

- **Gemini Strategy Engine**: Structured JSON prompts feed multi-symbol market context into Gemini, which returns actionable trade plans.
- **Risk Management**: Per-trade lot sizing via balance-based risk %, daily drawdown guard, concurrency caps, stop/TP buffers.
- **Autonomous Loop**: Background task refreshes data, requests Gemini analysis, and executes trades without manual intervention.
- **MT5 Integration**: Direct use of the official MetaTrader5 Python package for market data, tick feeds, and order placement.
- **Control Surface**: Live telemetry, signal log, open positions, and configuration management through a responsive web UI.

## Safety Switches

- `EXECUTION_ENABLED=false` in `.env` will run the logic without placing live orders.
- Confidence threshold (0.55 by default) prevents low-quality trades.
- Concurrency limit and daily drawdown guard enforce capital preservation.

## Project Scripts

- `npm run dev` – local Next.js dev server
- `npm run build` – production build
- `npm start` – serve production build
- `npm run lint` – lint sources
- `python python-bot/main.py` – start execution bridge

## Troubleshooting

- Ensure MT5 terminal is logged in and the account is funded (demo or live).
- If MT5 reports `TRADE_RETCODE_REQUOTE`, widen the request `deviation`.
- Gemini JSON parsing errors usually mean the model deviated from the schema; retry or adjust prompt settings.
- Update `MT5_SYMBOLS` and `MT5_TIMEFRAME` to align with your broker’s symbol names and available data feeds.

## Disclaimer

Automated trading carries significant financial risk. Use caution, validate strategies on demo accounts, and comply with local regulations before trading live capital.
