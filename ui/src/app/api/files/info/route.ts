import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { isAcceleratedApiRequestAuthenticated, resolveDownloadFileRequest } from '@/server/fileServing';
import { readSafetensorsInfo, SafetensorsInfoError } from '@/server/safetensorsInfo';

export const POST = withCommandBoundary(async (request: Request) => {
  if (!(await isAcceleratedApiRequestAuthenticated(request.headers))) return new Response('Unauthorized', { status: 401 });
  const { filePath } = await readJsonCommand(request);
  // Paths from FilesWidget are already decoded; percent characters are literal.
  if (typeof filePath !== 'string' || filePath.length > 4096 || !/\.safetensors$/i.test(filePath) || filePath.startsWith('remote://'))
    return new Response('A local safetensors filePath is required', { status: 400 });
  try {
    // Reuse canonical containment, configured storage roots and encrypted-dataset denial.
    const resolution = await resolveDownloadFileRequest(filePath, new Headers());
    if (resolution.kind !== 'response') return new Response('Unsupported file', { status: 400 });
    if (!resolution.file) return new Response(resolution.body, { status: resolution.status });
    return Response.json(await readSafetensorsInfo(resolution.file.path), { headers: { 'Cache-Control': 'no-store' } });
  } catch (error) {
    return new Response(error instanceof SafetensorsInfoError ? error.message : 'Cannot inspect file', {
      status: error instanceof SafetensorsInfoError ? 400 : 500,
    });
  }
});
