import { NextResponse } from 'next/server';
import { ensureDefaultAlertRules } from '@/server/alerts';
import { db } from '@/server/db';

export async function GET() {
  await ensureDefaultAlertRules();
  const rules = await db.alerts.listRules();
  return NextResponse.json({ rules });
}

export async function PATCH(request: Request) {
  const body = await request.json();
  if (!body.id || typeof body.id !== 'string') {
    return NextResponse.json({ error: 'Rule id is required' }, { status: 400 });
  }
  const rule = await db.alerts.updateRule(body.id, {
    enabled: typeof body.enabled === 'boolean' ? body.enabled : undefined,
    threshold: typeof body.threshold === 'number' ? body.threshold : undefined,
    severity: typeof body.severity === 'string' ? body.severity : undefined,
  } as any);
  return NextResponse.json(rule);
}
