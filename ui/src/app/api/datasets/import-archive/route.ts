import { NextRequest } from 'next/server';
import { acceptArchiveImport, readArchiveImport } from '@/server/archiveImportRoutes';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const POST = (request: NextRequest) => acceptArchiveImport(request, 'dataset-import');
export const GET = (request: NextRequest) => readArchiveImport(request, 'dataset-import');
