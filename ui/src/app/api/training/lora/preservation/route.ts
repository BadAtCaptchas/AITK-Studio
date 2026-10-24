import { createPreservationDataset } from '@/server/preservationDataset';
import { readJsonCommand } from '@/server/commandInput';

export async function POST(request: Request) {
  try {
    const value: unknown = await readJsonCommand(request, { maxBytes: 512 * 1024 });
    return Response.json(await createPreservationDataset(value));
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : 'Could not create preservation dataset.' }, { status: 400 });
  }
}
