import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import {
  buildUploadedLoraEntry,
  extractTriggerWordsFromMetadata,
  findDuplicateUploadedLoraFile,
  getUploadedLoraRoot,
  mergeTriggerWords,
  nextAvailableLoraPath,
  readSafetensorsMetadataStrict,
  splitTriggerWords,
  writeUploadedLoraSidecar,
} from '@/server/loraLibrary';
import {
  cleanupStagedUpload,
  decodedUploadHeader,
  InvalidUploadError,
  moveStagedUploadNoReplace,
  streamRequestToStagingFile,
} from '@/server/streamedUpload';

export const runtime = 'nodejs';

const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;

function isDestinationCollision(error: unknown) {
  return error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST';
}

export async function POST(request: NextRequest) {
  let savedPath: string | null = null;
  let createdPath: string | null = null;
  let stagingPath: string | null = null;

  try {
    const contentType = (request.headers.get('content-type') || '').split(';', 1)[0].trim().toLowerCase();
    if (contentType !== 'application/octet-stream') {
      return NextResponse.json({ error: 'LoRA uploads must use a streamed binary request' }, { status: 415 });
    }

    const originalFilename = decodedUploadHeader(request, 'x-aitk-file-name', 512);
    if (!originalFilename) {
      return NextResponse.json({ error: 'LoRA filename is required' }, { status: 400 });
    }
    if (!originalFilename.toLowerCase().endsWith('.safetensors')) {
      return NextResponse.json({ error: 'LoRA upload must be a .safetensors file' }, { status: 400 });
    }

    const root = await getUploadedLoraRoot();
    await fs.promises.mkdir(root, { recursive: true });
    const staged = await streamRequestToStagingFile(request, root, {
      maxBytes: MAX_FILE_BYTES,
      prefix: 'lora-upload',
    });
    const pendingUploadPath = staged.stagingPath;
    stagingPath = pendingUploadPath;
    const metadata = await readSafetensorsMetadataStrict(pendingUploadPath);
    let duplicatePath = await findDuplicateUploadedLoraFile(root, originalFilename, pendingUploadPath);
    let reused = duplicatePath !== null;
    if (duplicatePath) {
      savedPath = duplicatePath;
      await cleanupStagedUpload(stagingPath);
      stagingPath = null;
    } else {
      for (let attempt = 0; attempt < 1_000; attempt += 1) {
        savedPath = await nextAvailableLoraPath(root, originalFilename);
        try {
          await moveStagedUploadNoReplace(pendingUploadPath, savedPath);
          stagingPath = null;
          createdPath = savedPath;
          break;
        } catch (error) {
          if (!isDestinationCollision(error)) throw error;
          duplicatePath = await findDuplicateUploadedLoraFile(root, originalFilename, pendingUploadPath);
          if (duplicatePath) {
            savedPath = duplicatePath;
            reused = true;
            await cleanupStagedUpload(stagingPath);
            stagingPath = null;
            break;
          }
        }
      }
      if (stagingPath) {
        throw new InvalidUploadError('Could not reserve a unique LoRA filename');
      }
    }
    if (!savedPath) {
      throw new InvalidUploadError('LoRA upload did not produce a saved file');
    }

    const userTriggerWords = splitTriggerWords(decodedUploadHeader(request, 'x-aitk-trigger-words', 8_192));
    const metadataTriggerWords = extractTriggerWordsFromMetadata(metadata);
    const triggerWords = userTriggerWords.length > 0 ? userTriggerWords : metadataTriggerWords;

    if (!reused || userTriggerWords.length > 0) {
      await writeUploadedLoraSidecar(savedPath, {
        originalFilename,
        uploadedAt: new Date().toISOString(),
        triggerWords: mergeTriggerWords(triggerWords),
        triggerWordSource: userTriggerWords.length > 0 ? 'user' : metadataTriggerWords.length > 0 ? 'metadata' : 'none',
      });
    }

    const lora = await buildUploadedLoraEntry(savedPath);
    return NextResponse.json({ lora, reused });
  } catch (error) {
    await cleanupStagedUpload(stagingPath);
    if (createdPath) {
      await fs.promises.unlink(createdPath).catch(() => {});
    }

    console.error('LoRA upload error:', error);
    const message = error instanceof Error ? error.message : 'Error uploading LoRA';
    const status =
      error && typeof error === 'object' && 'status' in error && typeof error.status === 'number'
        ? error.status
        : 400;
    return NextResponse.json({ error: message }, { status });
  }
}
