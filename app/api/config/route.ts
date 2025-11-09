import { NextRequest, NextResponse } from 'next/server';
import { botConfigSchema, getBotState, setBotConfig } from '@/lib/state';

export async function GET() {
  const state = getBotState();
  return NextResponse.json(state.config, {
    headers: {
      'Cache-Control': 'no-store'
    }
  });
}

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const parsed = botConfigSchema.parse(payload);
    const state = setBotConfig(parsed);
    return NextResponse.json(state.config);
  } catch (error) {
    return NextResponse.json(
      {
        error: 'InvalidConfiguration',
        message:
          error instanceof Error
            ? error.message
            : 'Configuration payload failed validation.'
      },
      { status: 400 }
    );
  }
}
