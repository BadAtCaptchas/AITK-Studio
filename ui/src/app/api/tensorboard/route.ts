import { NextResponse } from 'next/server';
import { getTrainingFolder } from '@/server/settings';
import { getTensorBoardStatus } from '@/server/tensorboard';

export async function GET(request: Request) {
  try {
    const trainingRoot = await getTrainingFolder();
    const status = await getTensorBoardStatus(trainingRoot, request.url);
    return NextResponse.json(status);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch TensorBoard status' }, { status: 500 });
  }
}
