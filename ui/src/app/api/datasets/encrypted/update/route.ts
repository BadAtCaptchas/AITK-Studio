import { DATASET_MIGRATION_MESSAGE, DATASET_MIGRATION_REQUIRED } from '@/server/datasetDecryption';

function retired(): Response {
  return Response.json({ error: DATASET_MIGRATION_MESSAGE, code: DATASET_MIGRATION_REQUIRED }, { status: 410 });
}
export const POST = retired;
export const GET = retired;
