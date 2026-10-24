import { inspectContinuationSource } from '@/server/loraContinuation';
import { readJsonCommand } from '@/server/commandInput';

export async function POST(request: Request) {
  try {
    const value: unknown = await readJsonCommand(request, { maxBytes: 16 * 1024 });
    return Response.json(await inspectContinuationSource(value, request.signal));
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : 'LoRA inspection failed.' }, { status: 400 });
  }
}
