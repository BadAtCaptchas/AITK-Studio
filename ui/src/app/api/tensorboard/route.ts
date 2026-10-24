import { getTrainingFolder } from '@/server/settings';
import { getTensorBoardStatus } from '@/server/tensorboard';

export async function GET(request: Request) {
  try {
    const trainingRoot = await getTrainingFolder();
    const status = await getTensorBoardStatus(trainingRoot, request.url);
    return Response.json(status);
  } catch (error) {
    return Response.json({ error: 'Failed to fetch TensorBoard status' }, { status: 500 });
  }
}
