import { readWorkerDatasetCheck } from '@/server/trainingPreflight';
import { commandError, readJsonCommand } from '@/server/commandInput';

export async function POST(request: Request) {
  try {
    const value = await readJsonCommand(request, { maxBytes: 4096 });
    return Response.json(await readWorkerDatasetCheck(value));
  } catch (error) {
    const invalid = commandError(error);
    return Response.json({ error: invalid?.error || 'Dataset could not be checked on this worker' }, { status: invalid?.status || 400 });
  }
}
