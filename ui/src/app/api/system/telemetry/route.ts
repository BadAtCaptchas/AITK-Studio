import { NextResponse } from 'next/server';
import { listTelemetry, type TelemetryRange } from '@/server/systemTelemetry';

function cleanRange(value: string | null): TelemetryRange {
  if (value === '1h' || value === '6h' || value === '24h' || value === 'all') return value;
  return '6h';
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const range = cleanRange(searchParams.get('range'));
  const samples = await listTelemetry(range);
  return NextResponse.json({ range, samples });
}
