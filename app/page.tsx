"use client";

import { useState, type CSSProperties } from "react";
import useSWR from "swr";

type StatusResponse = {
  status: "idle" | "running" | "paused" | "error";
  accountBalance: number;
  equity: number;
  freeMargin: number;
  openPositions: Array<{
    id: string;
    symbol: string;
    direction: "buy" | "sell";
    entryPrice: number;
    stopLoss: number;
    takeProfit: number;
    riskR: number;
    openedAt: string;
  }>;
  recentSignals: Array<{
    id: string;
    symbol: string;
    timeframe: string;
    confidence: number;
    summary: string;
    generatedAt: string;
  }>;
  logs: Array<{
    id: string;
    level: "info" | "warn" | "error" | "success";
    message: string;
    timestamp: string;
  }>;
  config: {
    riskPerTrade: number;
    maxConcurrentTrades: number;
    maxDailyDrawdown: number;
    takeProfitRR: number;
    stopLossBufferPips: number;
    symbols: string[];
    enableNewsFilter: boolean;
    tradeSessionStart: string;
    tradeSessionEnd: string;
    enableAutoHedging: boolean;
    aggressionLevel: "conservative" | "balanced" | "aggressive";
  };
  lastHeartbeat: string;
};

const fetcher = (url: string) =>
  fetch(url, { cache: "no-store" }).then((res) => res.json());

export default function Home() {
  const { data, mutate, isLoading } = useSWR<StatusResponse>(
    "/api/status",
    fetcher,
    {
      refreshInterval: 5000,
    }
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  const handleConfigSave = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!data) return;

    const formData = new FormData(event.currentTarget);
    const payload = {
      riskPerTrade: Number(formData.get("riskPerTrade")),
      maxConcurrentTrades: Number(formData.get("maxConcurrentTrades")),
      maxDailyDrawdown: Number(formData.get("maxDailyDrawdown")),
      takeProfitRR: Number(formData.get("takeProfitRR")),
      stopLossBufferPips: Number(formData.get("stopLossBufferPips")),
      symbols: (formData.get("symbols") as string)
        .split(",")
        .map((sym) => sym.trim().toUpperCase())
        .filter(Boolean),
      enableNewsFilter: formData.get("enableNewsFilter") === "on",
      tradeSessionStart: formData.get("tradeSessionStart") as string,
      tradeSessionEnd: formData.get("tradeSessionEnd") as string,
      enableAutoHedging: formData.get("enableAutoHedging") === "on",
      aggressionLevel: formData.get("aggressionLevel") as
        | "conservative"
        | "balanced"
        | "aggressive",
    };

    setIsSubmitting(true);
    try {
      await fetch("/api/config", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      mutate();
    } catch (error) {
      console.error(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAnalysis = async () => {
    setIsRunning(true);
    try {
      await fetch("/api/analyze", {
        method: "POST",
      });
      mutate();
    } catch (error) {
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="grid">
      <section className="card">
        <div className="card-header">
          <h2>Trade Engine</h2>
          <span className="badge">
            {isLoading ? "syncing..." : data?.status ?? "unknown"}
          </span>
        </div>
        <div className="status-overview">
          <div>
            <strong>Balance</strong>
            <p>${(data?.accountBalance ?? 0).toFixed(2)}</p>
          </div>
          <div>
            <strong>Equity</strong>
            <p>${(data?.equity ?? 0).toFixed(2)}</p>
          </div>
          <div>
            <strong>Free Margin</strong>
            <p>${(data?.freeMargin ?? 0).toFixed(2)}</p>
          </div>
          <div>
            <strong>Heartbeat</strong>
            <p>
              {data
                ? new Intl.DateTimeFormat("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  }).format(new Date(data.lastHeartbeat))
                : "-"}
            </p>
          </div>
        </div>
        <div className="actions">
          <button
            className="primary"
            onClick={handleAnalysis}
            disabled={isRunning}
          >
            {isRunning ? "Running cycle..." : "Run analysis now"}
          </button>
          <button
            className="secondary"
            onClick={() => mutate()}
            disabled={isLoading}
          >
            Refresh state
          </button>
        </div>
        <div className="positions">
          <h3>Open Positions</h3>
          {data?.openPositions?.length ? (
            <ul className="status-list">
              {data.openPositions.map((position) => (
                <li key={position.id} className="status-item">
                  <div>
                    <strong>
                      {position.symbol} · {position.direction.toUpperCase()}
                    </strong>
                  </div>
                  <span>
                    Entry {position.entryPrice.toFixed(5)} · SL{" "}
                    {position.stopLoss.toFixed(5)} · TP{" "}
                    {position.takeProfit.toFixed(5)}
                  </span>
                  <span>Risk {position.riskR.toFixed(2)}R</span>
                  <span>
                    Opened{" "}
                    {new Intl.DateTimeFormat("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    }).format(new Date(position.openedAt))}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p>No live positions.</p>
          )}
        </div>
        <div className="signals">
          <h3>AI Signals</h3>
          {data?.recentSignals?.length ? (
            <ul className="status-list">
              {data.recentSignals.map((signal) => (
                <li key={signal.id} className="status-item">
                  <div>
                    <strong>
                      {signal.symbol} · {signal.timeframe}
                    </strong>
                  </div>
                  <span>
                    Confidence {(signal.confidence * 100).toFixed(1)}%
                  </span>
                  <span>{signal.summary}</span>
                  <span>
                    {new Intl.DateTimeFormat("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    }).format(new Date(signal.generatedAt))}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p>No signal history yet.</p>
          )}
        </div>
      </section>

      <section className="card">
        <h2>Execution Settings</h2>
        <form className="input-grid" onSubmit={handleConfigSave}>
          <label>
            Risk Per Trade (%)
            <input
              name="riskPerTrade"
              type="number"
              step="0.1"
              defaultValue={data?.config.riskPerTrade ?? 1}
            />
          </label>
          <label>
            Max Concurrent Trades
            <input
              name="maxConcurrentTrades"
              type="number"
              step="1"
              defaultValue={data?.config.maxConcurrentTrades ?? 3}
            />
          </label>
          <label>
            Max Daily Drawdown (%)
            <input
              name="maxDailyDrawdown"
              type="number"
              step="0.1"
              defaultValue={data?.config.maxDailyDrawdown ?? 5}
            />
          </label>
          <label>
            Target RR
            <input
              name="takeProfitRR"
              type="number"
              step="0.1"
              defaultValue={data?.config.takeProfitRR ?? 2}
            />
          </label>
          <label>
            Stop Loss Buffer (pips)
            <input
              name="stopLossBufferPips"
              type="number"
              step="0.1"
              defaultValue={data?.config.stopLossBufferPips ?? 5}
            />
          </label>
          <label>
            Symbols (comma separated)
            <input
              name="symbols"
              type="text"
              defaultValue={
                data?.config.symbols?.join(", ") ?? "EURUSD"
              }
            />
          </label>
          <label>
            Session Start
            <input
              name="tradeSessionStart"
              type="time"
              defaultValue={data?.config.tradeSessionStart ?? "07:00"}
            />
          </label>
          <label>
            Session End
            <input
              name="tradeSessionEnd"
              type="time"
              defaultValue={data?.config.tradeSessionEnd ?? "20:00"}
            />
          </label>
          <label>
            Aggression Level
            <select
              name="aggressionLevel"
              defaultValue={data?.config.aggressionLevel ?? "balanced"}
            >
              <option value="conservative">Conservative</option>
              <option value="balanced">Balanced</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </label>
          <label>
            <div className="checkbox">
              <input
                name="enableNewsFilter"
                type="checkbox"
                defaultChecked={data?.config.enableNewsFilter ?? true}
              />
              3rd-Party News Filter
            </div>
          </label>
          <label>
            <div className="checkbox">
              <input
                name="enableAutoHedging"
                type="checkbox"
                defaultChecked={data?.config.enableAutoHedging ?? false}
              />
              Auto Hedging
            </div>
          </label>
          <button
            className="primary"
            type="submit"
            style={{ gridColumn: "span 2" } as CSSProperties}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Saving..." : "Save configuration"}
          </button>
        </form>
      </section>

      <section className="card" style={{ gridColumn: "span 2" }}>
        <h2>Event Log</h2>
        <div className="log-stream">
          {data?.logs?.length ? (
            data.logs.map((log) => (
              <div key={log.id} className={`log-entry ${log.level}`}>
                [{new Intl.DateTimeFormat("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                }).format(new Date(log.timestamp))}] {log.message}
              </div>
            ))
          ) : (
            <p>No logs available yet.</p>
          )}
        </div>
      </section>
    </div>
  );
}
