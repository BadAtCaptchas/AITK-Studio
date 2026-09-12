import { NextResponse } from 'next/server';
import { getReadiness } from '@/server/readiness';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export async function GET() {
  try {
    return NextResponse.json(await getReadiness(), { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return NextResponse.json(
      { error: 'Diagnostics unavailable. Check database and worker startup logs.' },
      { status: 503 },
    );
  }
}
