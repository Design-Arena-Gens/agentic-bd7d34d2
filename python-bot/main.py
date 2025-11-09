import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

import httpx
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "google-generativeai package is required. Install deps via pip install -r requirements.txt"
    ) from exc


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mt5-bot")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradePlan(BaseModel):
    symbol: str
    bias: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0, le=1)
    entry: float
    stopLoss: float
    takeProfit: float
    rationale: str
    riskToReward: float
    positionSizing: Dict[str, float]


class ExecutionResult(BaseModel):
    symbol: str
    orderId: Optional[int]
    lots: float
    entryPrice: float
    stopLoss: float
    takeProfit: float
    status: Literal["placed", "skipped", "failed"]
    reason: Optional[str] = None


class BotConfig(BaseModel):
    terminal_path: str = Field(
        default=os.environ.get(
            "MT5_TERMINAL_PATH", r"G:\Program Files\terminal64\terminal64.exe"
        )
    )
    login: Optional[int] = (
        int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None
    )
    password: Optional[str] = os.environ.get("MT5_PASSWORD")
    server: Optional[str] = os.environ.get("MT5_SERVER")
    symbols: List[str] = Field(
        default_factory=lambda: [
            sym.strip().upper()
            for sym in os.environ.get("MT5_SYMBOLS", "EURUSD,GBPUSD,XAUUSD").split(",")
            if sym.strip()
        ]
    )
    timeframe: str = os.environ.get("MT5_TIMEFRAME", "M15")
    risk_per_trade_percent: float = float(
        os.environ.get("RISK_PER_TRADE_PERCENT", 1.0)
    )
    max_concurrent_trades: int = int(
        os.environ.get("MAX_CONCURRENT_TRADES", 3)
    )
    max_daily_drawdown_percent: float = float(
        os.environ.get("MAX_DAILY_DRAWDOWN_PERCENT", 5.0)
    )
    stop_loss_buffer_pips: float = float(
        os.environ.get("STOP_LOSS_BUFFER_PIPS", 5.0)
    )
    take_profit_rr: float = float(os.environ.get("TAKE_PROFIT_RR", 2.0))
    analysis_interval_seconds: int = int(
        os.environ.get("ANALYSIS_INTERVAL_SECONDS", 120)
    )
    control_plane_url: str = os.environ.get("CONTROL_PLANE_URL", "http://localhost:3000")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    execution_enabled: bool = os.environ.get("EXECUTION_ENABLED", "true").lower() in [
        "1",
        "true",
        "yes",
    ]

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        normalized = value.upper().strip()
        if not normalized.startswith("M") and not normalized.startswith("H") and not normalized.startswith("D"):
            raise ValueError("Timeframe must be M1, M5, M15, H1, etc.")
        return normalized


config = BotConfig()


class GeminiPlanner:
    def __init__(self) -> None:
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=0.25,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
            system_instruction=(
                "You are an institutional forex strategist. "
                "Only output valid JSON that matches the requested schema. "
                "Assess supply/demand, trend alignment, liquidity sweeps, and "
                "probability of success. Avoid trades with poor reward-to-risk."
            ),
        )

    async def plan(
        self,
        market_snapshots: List[Dict],
        risk_profile: Dict,
    ) -> Dict:
        payload = {
            "marketSnapshots": market_snapshots,
            "riskProfile": risk_profile,
            "instructions": {
                "outputShape": {
                    "generatedAt": "ISO timestamp",
                    "plans": [
                        {
                            "symbol": "string",
                            "bias": "long|short|flat",
                            "confidence": "0-1 float",
                            "entry": "float",
                            "stopLoss": "float",
                            "takeProfit": "float",
                            "rationale": "string",
                            "riskToReward": "float",
                            "positionSizing": {
                                "lots": "float",
                                "riskPercent": "float",
                            },
                        }
                    ],
                    "riskAdvisory": "string",
                }
            },
        }

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                contents=[{"role": "user", "parts": [json.dumps(payload)]}]
            ),
        )

        text = (
            response.candidates[0]
            .content.parts[0]
            .text.strip()
            if response.candidates
            else ""
        )

        if not text:
            raise RuntimeError("Gemini returned empty payload.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini responded with invalid JSON: {text}") from exc


class EventEmitter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=10)

    async def _post(self, path: str, payload: Dict) -> None:
        try:
            url = f"{self.base_url}{path}"
            await self.client.post(url, json=payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to emit event to control plane: %s", exc)

    async def log(self, level: str, message: str) -> None:
        await self._post(
            "/api/events",
            {"type": "log", "level": level, "message": message},
        )

    async def state(self, payload: Dict) -> None:
        await self._post("/api/events", {"type": "state", "payload": payload})

    async def signal(self, payload: Dict) -> None:
        await self._post("/api/events", {"type": "signal", **payload})

    async def position_open(self, payload: Dict) -> None:
        await self._post("/api/events", {"type": "position_open", **payload})

    async def position_close(self, payload: Dict) -> None:
        await self._post("/api/events", {"type": "position_close", **payload})


event_emitter = EventEmitter(config.control_plane_url)
gemini_planner = GeminiPlanner()


def timeframe_to_mt5(timeframe: str) -> int:
    tf = timeframe.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe {tf}")
    return mapping[tf]


class Mt5Bridge:
    def __init__(self) -> None:
        self.timeframe = timeframe_to_mt5(config.timeframe)
        self.lock = asyncio.Lock()
        self._ensure_terminal()
        self._initialize()

    def _ensure_terminal(self) -> None:
        path = config.terminal_path
        if os.name == "nt" and path and os.path.exists(path):
            logger.info("Launching MetaTrader terminal from %s", path)
            subprocess.Popen([path], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # allow terminal to boot
            time.sleep(5)
        else:
            logger.warning("MT5 terminal path invalid or not on Windows; skipping launch.")

    def _initialize(self) -> None:
        logger.info("Initializing MetaTrader5 bridge.")
        initialized = mt5.initialize(
            login=config.login,
            password=config.password,
            server=config.server,
            path=config.terminal_path if os.path.exists(config.terminal_path) else None,
        )
        if not initialized:
            error = mt5.last_error()
            raise RuntimeError(f"Failed to initialize MT5: {error}")
        logger.info("MT5 initialized successfully.")

    async def refresh_account_metrics(self) -> Dict:
        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("Unable to fetch account info.")
        return {
            "balance": float(account_info.balance),
            "equity": float(account_info.equity),
            "marginFree": float(account_info.margin_free),
        }

    async def collect_market_snapshot(self) -> List[Dict]:
        snapshots: List[Dict] = []
        for symbol in config.symbols:
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, 300)
            if rates is None:
                logger.warning("No data returned for %s", symbol)
                continue

            df = pd.DataFrame(rates)
            if df.empty:
                continue

            df["ema_fast"] = df["close"].ewm(span=20).mean()
            df["ema_slow"] = df["close"].ewm(span=50).mean()
            df["rsi"] = compute_rsi(df["close"], period=14)
            df["atr"] = compute_atr(df, period=14)
            df["volatility"] = df["close"].pct_change().rolling(window=30).std() * np.sqrt(252)
            df = df.fillna(method="ffill").fillna(method="bfill")

            sentiment = ((df["close"].iloc[-1] - df["ema_slow"].iloc[-1]) / df["atr"].iloc[-1]) if df["atr"].iloc[-1] else 0
            info = mt5.symbol_info(symbol)
            spread = float(info.spread * info.point) if info else 0.0

            snapshots.append(
                {
                    "symbol": symbol,
                    "timeframe": config.timeframe,
                    "candles": df.tail(120)[["time", "open", "high", "low", "close", "tick_volume"]]
                    .rename(columns={"tick_volume": "volume"})
                    .to_dict(orient="records"),
                    "indicators": {
                        "emaFast": float(df["ema_fast"].iloc[-1]),
                        "emaSlow": float(df["ema_slow"].iloc[-1]),
                        "rsi14": float(df["rsi"].iloc[-1]),
                        "atr14": float(df["atr"].iloc[-1]),
                        "volatility": float(df["volatility"].iloc[-1]),
                    },
                    "sentimentScore": float(np.tanh(sentiment)),
                    "spread": spread,
                }
            )
        return snapshots

    async def can_open_trade(self) -> bool:
        positions = mt5.positions_get()
        return positions is None or len(positions) < config.max_concurrent_trades

    async def place_order(
        self,
        symbol: str,
        direction: Literal["long", "short"],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lots: float,
    ) -> Tuple[Optional[int], Optional[str]]:
        if not config.execution_enabled:
            return None, "Execution disabled by configuration."

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None, "No tick data available."

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(lots, 2),
            "type": order_type,
            "price": tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 352719,
            "comment": "Gemini auto-trader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            return None, "Order send failed with no result."

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            reason = f"Order failed: {result.comment} ({result.retcode})"
            return None, reason

        return result.order, None


mt5_bridge = Mt5Bridge()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


async def size_position(symbol: str, entry: float, stop: float, balance: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0

    point = info.point
    tick_value = info.trade_tick_value
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0.0

    risk_amount = balance * (config.risk_per_trade_percent / 100)
    value_per_point = tick_value / info.trade_tick_size if info.trade_tick_size else tick_value
    stop_points = stop_distance / point
    lots = risk_amount / (stop_points * value_per_point)
    volume_min = info.volume_min if info.volume_min else 0.01
    volume_step = info.volume_step if info.volume_step else 0.01
    lots = max(volume_min, min(info.volume_max, lots))
    lots = round(lots / volume_step) * volume_step
    return float(round(lots, 2))


async def run_cycle() -> List[ExecutionResult]:
    async with mt5_bridge.lock:
        account_metrics = await mt5_bridge.refresh_account_metrics()
        await event_emitter.state(
            {
                "status": "running",
                "accountBalance": account_metrics["balance"],
                "equity": account_metrics["equity"],
                "freeMargin": account_metrics["marginFree"],
            }
        )

        snapshots = await mt5_bridge.collect_market_snapshot()
        if not snapshots:
            raise RuntimeError("No market snapshots available.")

        risk_profile = {
            "balance": account_metrics["balance"],
            "riskPerTrade": config.risk_per_trade_percent,
            "maxDailyDrawdown": config.max_daily_drawdown_percent,
            "aggressionLevel": "balanced",
        }

        gemini_plan = await gemini_planner.plan(snapshots, risk_profile)

        plans = [TradePlan(**plan) for plan in gemini_plan.get("plans", [])]
        if not plans:
            raise RuntimeError("Gemini did not return any actionable plans.")

        execution_results: List[ExecutionResult] = []

        for plan in plans:
            await event_emitter.signal(
                {
                    "symbol": plan.symbol,
                    "timeframe": config.timeframe,
                    "confidence": plan.confidence,
                    "summary": plan.rationale,
                }
            )

            if plan.bias == "flat" or plan.confidence < 0.55:
                execution_results.append(
                    ExecutionResult(
                        symbol=plan.symbol,
                        orderId=None,
                        lots=0,
                        entryPrice=plan.entry,
                        stopLoss=plan.stopLoss,
                        takeProfit=plan.takeProfit,
                        status="skipped",
                        reason="Confidence below threshold or flat bias.",
                    )
                )
                continue

            if not await mt5_bridge.can_open_trade():
                execution_results.append(
                    ExecutionResult(
                        symbol=plan.symbol,
                        orderId=None,
                        lots=0,
                        entryPrice=plan.entry,
                        stopLoss=plan.stopLoss,
                        takeProfit=plan.takeProfit,
                        status="skipped",
                        reason="Max concurrent trades reached.",
                    )
                )
                continue

            lots = await size_position(
                plan.symbol, plan.entry, plan.stopLoss, account_metrics["balance"]
            )
            if lots <= 0:
                execution_results.append(
                    ExecutionResult(
                        symbol=plan.symbol,
                        orderId=None,
                        lots=0,
                        entryPrice=plan.entry,
                        stopLoss=plan.stopLoss,
                        takeProfit=plan.takeProfit,
                        status="skipped",
                        reason="Unable to compute valid lot size.",
                    )
                )
                continue

            order_id, error = await mt5_bridge.place_order(
                symbol=plan.symbol,
                direction="long" if plan.bias == "long" else "short",
                entry_price=plan.entry,
                stop_loss=plan.stopLoss,
                take_profit=plan.takeProfit,
                lots=lots,
            )

            if error:
                execution_results.append(
                    ExecutionResult(
                        symbol=plan.symbol,
                        orderId=None,
                        lots=lots,
                        entryPrice=plan.entry,
                        stopLoss=plan.stopLoss,
                        takeProfit=plan.takeProfit,
                        status="failed",
                        reason=error,
                    )
                )
                await event_emitter.log("error", f"{plan.symbol} execution failed: {error}")
            else:
                execution_results.append(
                    ExecutionResult(
                        symbol=plan.symbol,
                        orderId=order_id,
                        lots=lots,
                        entryPrice=plan.entry,
                        stopLoss=plan.stopLoss,
                        takeProfit=plan.takeProfit,
                        status="placed",
                    )
                )
                await event_emitter.position_open(
                    {
                        "symbol": plan.symbol,
                        "direction": "buy" if plan.bias == "long" else "sell",
                        "entryPrice": plan.entry,
                        "stopLoss": plan.stopLoss,
                        "takeProfit": plan.takeProfit,
                        "riskR": plan.riskToReward,
                    }
                )

        await event_emitter.log(
            "info",
            f"Completed cycle with {len(execution_results)} opportunities.",
        )

        return execution_results


app = FastAPI(
    title="Gemini MT5 Execution Bridge",
    description="HTTP bridge that brokers communication between MetaTrader5 and the web control plane.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExecutePayload(BaseModel):
    plans: List[TradePlan]


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(cycle_worker())
    await event_emitter.log("info", "MT5 bridge online.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await event_emitter.log("warn", "MT5 bridge shutting down.")
    await event_emitter.client.aclose()
    mt5.shutdown()


@app.get("/health")
async def health() -> Dict:
    terminal = mt5.terminal_info()
    initialized = terminal is not None
    return {
        "status": "ok" if initialized else "error",
        "time": iso_now(),
    }


@app.get("/config")
async def get_config() -> Dict:
    return config.model_dump()


@app.get("/market/snapshot")
async def market_snapshot() -> List[Dict]:
    return await mt5_bridge.collect_market_snapshot()


@app.post("/trades/execute")
async def execute_trades(payload: ExecutePayload) -> List[ExecutionResult]:
    results: List[ExecutionResult] = []
    account_metrics = await mt5_bridge.refresh_account_metrics()
    for plan in payload.plans:
        lots = await size_position(
            plan.symbol, plan.entry, plan.stopLoss, account_metrics["balance"]
        )
        order_id, error = await mt5_bridge.place_order(
            symbol=plan.symbol,
            direction="long" if plan.bias == "long" else "short",
            entry_price=plan.entry,
            stop_loss=plan.stopLoss,
            take_profit=plan.takeProfit,
            lots=lots,
        )
        status: Literal["placed", "failed"] = "placed" if not error else "failed"
        results.append(
            ExecutionResult(
                symbol=plan.symbol,
                orderId=order_id,
                lots=lots,
                entryPrice=plan.entry,
                stopLoss=plan.stopLoss,
                takeProfit=plan.takeProfit,
                status=status,
                reason=error,
            )
        )
    return results


async def cycle_worker() -> None:
    while True:
        try:
            await run_cycle()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Cycle failed: %s", exc)
            await event_emitter.log("error", f"Cycle failure: {exc}")
            await event_emitter.state({"status": "error"})
        await asyncio.sleep(config.analysis_interval_seconds)


def main() -> None:
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8770,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
