import { NextResponse } from 'next/server';
import { ensureDefaultAlertRules } from '@/server/alerts';
import { db } from '@/server/db';

export async function GET(request: Request) {
  await ensureDefaultAlertRules();
  const { searchParams } = new URL(request.url);
  const status = searchParams.get('status') || undefined;
  const events = await db.alerts.listEvents({ status, limit: 500 });
  return NextResponse.json({ events });
}
