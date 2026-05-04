import { NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const { id } = await params;
  const body = await request.json();
  const patch: any = {};
  if (typeof body.status === 'string') patch.status = body.status;
  if (body.status === 'acknowledged') patch.acknowledged_at = new Date();
  if (body.status === 'resolved') patch.resolved_at = new Date();
  if (typeof body.snoozed_until === 'string') patch.snoozed_until = new Date(body.snoozed_until);
  const event = await db.alerts.updateEvent(id, patch);
  return NextResponse.json(event);
}
