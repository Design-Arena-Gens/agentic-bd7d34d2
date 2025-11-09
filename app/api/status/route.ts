import { NextResponse } from 'next/server';
import { getBotState } from '@/lib/state';

export async function GET() {
  const state = getBotState();
  return NextResponse.json(state, {
    headers: {
      'Cache-Control': 'no-store'
    }
  });
}
