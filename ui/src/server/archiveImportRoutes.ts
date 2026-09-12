import { randomUUID } from 'crypto';
import {
  archiveUploadMode,
  cleanupOldArchiveUploadChunks,
  readArchiveUploadChunksTotal,
  readArchiveUploadFileBytes,
  readArchiveUploadID,
  saveArchiveUploadChunk,
} from './archiveUploadChunks';
import { receiveManifestChunk, getUploadManifest } from './archiveUploadManifest';
import { importRoot, MAX_IMPORT_BYTES, type ArchiveInput } from './archiveImports';
import { enqueueOperation, getOperation, operationID, type Operation } from './operations';
import { isRecord } from './commandInput';

type ArchiveRequest = Request & { nextUrl: URL };
export function archiveOperationStatus(operation: Operation) {
  const uploadID = operation.input.kind === 'remote-start' ? '' : operation.input.uploadID;
  return {
    uploadID,
    operationID: operation.id,
    status:
      operation.state === 'completed'
        ? 'completed'
        : ['failed', 'canceled'].includes(operation.state)
          ? 'failed'
          : 'importing',
    phase: operation.phase,
    state: operation.state,
    result: operation.result,
    error: operation.error,
    createdAt: operation.createdAt,
    updatedAt: operation.updatedAt,
  };
}
function failure(error: unknown): Response {
  const status = isRecord(error) && typeof error.status === 'number' ? error.status : 400;
  return Response.json({ error: error instanceof Error ? error.message : 'Archive request failed' }, { status });
}
export async function readArchiveImport(request: ArchiveRequest, kind: ArchiveInput['kind']): Promise<Response> {
  try {
    const operation = await getOperation(operationID(kind, readArchiveUploadID(request)));
    if (!operation) {
      const upload = await getUploadManifest(await importRoot(kind), readArchiveUploadID(request));
      if (!upload) return Response.json({ error: 'Upload not found' }, { status: 404 });
      return Response.json({
        uploadID: upload.uploadID,
        state: upload.state,
        receivedBytes: upload.receivedBytes,
        chunks: upload.chunks,
        expiresAt: upload.expiresAt,
      });
    }
    return Response.json(archiveOperationStatus(operation));
  } catch (error) {
    return failure(error);
  }
}
export async function acceptArchiveImport(request: ArchiveRequest, kind: ArchiveInput['kind']): Promise<Response> {
  try {
    const root = await importRoot(kind),
      mode = archiveUploadMode(request);
    if (mode === 'chunk') {
      await cleanupOldArchiveUploadChunks(root);
      return Response.json(await saveArchiveUploadChunk(request, root, { maxArchiveBytes: MAX_IMPORT_BYTES }));
    }
    const chunked = mode === 'complete';
    const uploadID =
      chunked || request.nextUrl.searchParams.has('uploadID') ? readArchiveUploadID(request) : randomUUID();
    const prior = await getOperation(operationID(kind, uploadID));
    if (prior) return Response.json(archiveOperationStatus(prior), { status: prior.state === 'completed' ? 200 : 202 });
    const input: ArchiveInput = {
      kind,
      root,
      uploadID,
      chunked,
      chunksTotal: chunked ? readArchiveUploadChunksTotal(request) : 1,
      expectedBytes: chunked ? readArchiveUploadFileBytes(request, MAX_IMPORT_BYTES) : null,
      preferredName: request.nextUrl.searchParams.get('preferredName') || request.headers.get('x-aitk-preferred-name'),
      gpuIds: request.nextUrl.searchParams.get('gpu_ids') || request.headers.get('x-aitk-gpu-ids'),
    };
    if (chunked) {
      const manifest = await getUploadManifest(root, uploadID);
      if (
        !manifest ||
        manifest.chunksTotal !== input.chunksTotal ||
        Object.keys(manifest.chunks).length !== input.chunksTotal
      )
        throw new Error('Archive upload is incomplete');
      if (input.expectedBytes !== null && input.expectedBytes !== manifest.receivedBytes)
        throw new Error('Invalid archive upload fileBytes');
    } else {
      if (
        !['application/octet-stream', 'application/zip', 'application/x-zip-compressed'].includes(
          (request.headers.get('content-type') || '').split(';')[0],
        )
      )
        return Response.json({ error: 'Archives require a streamed binary request' }, { status: 415 });
      await receiveManifestChunk(request, root, {
        uploadID,
        chunkIndex: 0,
        chunksTotal: 1,
        expectedBytes: null,
        limit: MAX_IMPORT_BYTES,
        purpose: request.nextUrl.pathname,
      });
      input.chunked = true;
    }

    const operation = await enqueueOperation(input, uploadID);
    return Response.json(archiveOperationStatus(operation), { status: 202 });
  } catch (error) {
    return failure(error);
  }
}
