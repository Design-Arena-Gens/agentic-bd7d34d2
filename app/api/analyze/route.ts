import { NextResponse } from 'next/server';
import { runAnalysisCycle } from '@/lib/analyzer';

export async function POST() {
  try {
    const result = await runAnalysisCycle();
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      {
        error: 'AnalysisFailed',
        message:
          error instanceof Error
            ? error.message
            : 'Gemini analysis cycle failed.'
      },
      { status: 500 }
    );
  }
}
