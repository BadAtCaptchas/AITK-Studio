import { fileResponse } from '@/server/http/fileResponse';
import {
  isAcceleratedApiRequestAuthenticated,
  resolveDownloadFileRequest,
  unauthorizedFileResponse,
} from '@/server/fileServing';



async function serve(
  request: Request,
  { params }: { params: Promise<{ filePath: string[] }> },
) {
  try {
    if (!(await isAcceleratedApiRequestAuthenticated(request.headers))) {
      return fileResponse(request, unauthorizedFileResponse());
    }
    if (request.signal.aborted) return new Response(null, { status: 499 });

    const { filePath } = await params;
    const decodedFilePath = decodeURIComponent(filePath.join('/'));
    return fileResponse(
      request,
      await resolveDownloadFileRequest(decodedFilePath, request.headers),
    );
  } catch (error) {
    console.error('Error serving file:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}

export const GET = serve;
export const HEAD = serve;
