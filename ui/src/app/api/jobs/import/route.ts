import { acceptArchiveImport, readArchiveImport } from '@/server/archiveImportRoutes';

export const POST = (request: Request) => acceptArchiveImport(request, 'job-import');
export const GET = (request: Request) => readArchiveImport(request, 'job-import');
