import { NextResponse } from 'next/server';
import { readWorkerDatasetCheck } from '@/server/trainingPreflight';
import { commandError, readJsonCommand } from '@/server/commandInput';

export const runtime = 'nodejs';
export async function POST(request: Request) {
  try {
    const value = await readJsonCommand(request, { maxBytes: 4096 });
    return NextResponse.json(await readWorkerDatasetCheck(value));
  } catch (error) {
    const invalid = commandError(error);
    return NextResponse.json({ error: invalid?.error || 'Dataset could not be checked on this worker' }, { status: invalid?.status || 400 });
  }
}
