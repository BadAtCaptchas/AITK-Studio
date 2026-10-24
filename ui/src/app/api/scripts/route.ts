import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { getTrainingFolder } from '@/server/settings';
import {
  buildScriptInvocation,
  runScriptBuffered,
  runScriptStreaming,
  ScriptValidationError,
} from '@/server/scriptRunner';

export const maxDuration = 1200;

async function postCommand(request: Request) {
  let body: unknown;
  try {
    body = assertGlobalPayload(await readJsonCommand(request));
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  let invocation;
  try {
    invocation = buildScriptInvocation(body, await getTrainingFolder());
  } catch (error) {
    if (error instanceof ScriptValidationError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    console.error('Error validating script request:', error);
    return Response.json({ error: 'Error validating script request' }, { status: 500 });
  }

  if (body && typeof body === 'object' && (body as Record<string, unknown>).stream === true) {
    return runScriptStreaming(invocation);
  }

  const result = await runScriptBuffered(invocation);
  const status = result.ok ? 200 : result.timedOut ? 504 : 500;
  return Response.json(result, { status });
}

export const POST = withCommandBoundary(postCommand);
