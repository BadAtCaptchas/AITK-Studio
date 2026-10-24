import { acceptArchiveImport, readArchiveImport } from '@/server/archiveImportRoutes';

export const POST = (request: Request) => acceptArchiveImport(request, 'dataset-import');
export const GET = (request: Request) => readArchiveImport(request, 'dataset-import');
