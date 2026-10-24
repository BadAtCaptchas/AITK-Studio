import fs from 'fs';
import { Readable } from 'stream';
import { getRemoteWorker, remoteJson, fetchWorkerHealth } from './remoteClient';
import { getUploadedLoraRoot, readSafetensorsMetadataStrict } from './loraLibrary';
import { cleanupStagedUpload, streamRequestToStagingFile } from './streamedUpload';
import { hashContinuationAsset, MAX_CONTINUATION_BYTES } from './loraContinuation';
import { isRecord } from './commandInput';
import { makeRemoteLoraRef } from '../utils/remoteLoraRefs';

export async function uploadLoraToWorker(request: Request, workerID: string): Promise<unknown> {
  const worker = await getRemoteWorker(workerID);
  if ((await fetchWorkerHealth(worker)).loraContinuationVersion !== 1) throw new Error('Update the worker before uploading a continuation LoRA.');
  const root = await getUploadedLoraRoot();
  await fs.promises.mkdir(root, { recursive: true });
  const staged = await streamRequestToStagingFile(request, root, { maxBytes: MAX_CONTINUATION_BYTES, prefix: 'remote-lora-upload' });
  let stream: fs.ReadStream | undefined;
  try {
    await readSafetensorsMetadataStrict(staged.stagingPath);
    const sha256 = await hashContinuationAsset(staged.stagingPath);
    const headers = new Headers({ 'Content-Type': 'application/octet-stream', 'X-AITK-Expected-SHA256': sha256 });
    for (const key of ['x-aitk-file-name', 'x-aitk-trigger-words']) {
      const value = request.headers.get(key);
      if (value) headers.set(key, value);
    }
    headers.set('Content-Length', String((await fs.promises.stat(staged.stagingPath)).size));
    stream = fs.createReadStream(staged.stagingPath);
    const init = { method: 'POST', headers, body: Readable.toWeb(stream) as ReadableStream<Uint8Array>,
      duplex: 'half', signal: request.signal, timeoutMs: 30 * 60_000, headerTimeoutMs: 30 * 60_000 };
    const result: unknown = await remoteJson(worker, '/api/generate/loras/upload', init);
    if (!isRecord(result) || !isRecord(result.lora) || typeof result.lora.path !== 'string' || result.sha256 !== sha256)
      throw new Error('Worker upload did not acknowledge the source checksum.');
    return { ...result, lora: { ...result.lora, path: makeRemoteLoraRef(workerID, result.lora.path) } };
  } finally {
    if (stream && !stream.closed) {
      // destroy() closes asynchronously, including when the file is still
      // opening. Windows cannot unlink the staged file until close completes.
      const closed = new Promise<void>(resolve => stream!.once('close', resolve));
      stream.destroy();
      await closed;
    }
    await cleanupStagedUpload(staged.stagingPath);
  }
}
