import { NextRequest, NextResponse } from 'next/server';
import {
  appendLog,
  closePosition,
  pushPosition,
  pushSignal,
  updateBotState
} from '@/lib/state';
import { z } from 'zod';

const eventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('log'),
    level: z.enum(['info', 'warn', 'error', 'success']).default('info'),
    message: z.string().min(2)
  }),
  z.object({
    type: z.literal('signal'),
    symbol: z.string(),
    timeframe: z.string(),
    confidence: z.number().min(0).max(1),
    summary: z.string().min(6)
  }),
  z.object({
    type: z.literal('position_open'),
    symbol: z.string(),
    direction: z.enum(['buy', 'sell']),
    entryPrice: z.number().positive(),
    stopLoss: z.number().positive(),
    takeProfit: z.number().positive(),
    riskR: z.number().min(0).max(10)
  }),
  z.object({
    type: z.literal('position_close'),
    positionId: z.string(),
    reason: z.string().optional()
  }),
  z.object({
    type: z.literal('state'),
    payload: z
      .object({
        status: z.enum(['idle', 'running', 'paused', 'error']).optional(),
        accountBalance: z.number().optional(),
        equity: z.number().optional(),
        freeMargin: z.number().optional()
      })
      .passthrough()
  })
]);

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const event = eventSchema.parse(body);

    switch (event.type) {
      case 'log':
        appendLog({ level: event.level, message: event.message });
        break;
      case 'signal':
        pushSignal({
          symbol: event.symbol,
          timeframe: event.timeframe,
          confidence: event.confidence,
          summary: event.summary
        });
        break;
      case 'position_open':
        pushPosition({
          symbol: event.symbol,
          direction: event.direction,
          entryPrice: event.entryPrice,
          stopLoss: event.stopLoss,
          takeProfit: event.takeProfit,
          riskR: event.riskR
        });
        break;
      case 'position_close':
        closePosition(event.positionId);
        appendLog({
          level: 'info',
          message: `Position ${event.positionId} closed${event.reason ? `: ${event.reason}` : ''}`
        });
        break;
      case 'state':
        updateBotState(event.payload);
        break;
      default:
        break;
    }

    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      {
        error: 'InvalidEvent',
        message:
          error instanceof Error
            ? error.message
            : 'Event payload failed validation.'
      },
      { status: 400 }
    );
  }
}
