import { NextResponse } from 'next/server';
import { getTrainingFolder } from '@/server/settings';
import {
  buildScriptInvocation,
  runScriptBuffered,
  runScriptStreaming,
  ScriptValidationError,
} from '@/server/scriptRunner';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 1200;

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  let invocation;
  try {
    invocation = buildScriptInvocation(body, await getTrainingFolder());
  } catch (error) {
    if (error instanceof ScriptValidationError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    console.error('Error validating script request:', error);
    return NextResponse.json({ error: 'Error validating script request' }, { status: 500 });
  }

  if (body && typeof body === 'object' && (body as Record<string, unknown>).stream === true) {
    return runScriptStreaming(invocation);
  }

  const result = await runScriptBuffered(invocation);
  const status = result.ok ? 200 : result.timedOut ? 504 : 500;
  return NextResponse.json(result, { status });
}
