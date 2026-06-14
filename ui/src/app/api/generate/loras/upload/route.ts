import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import {
  buildUploadedLoraEntry,
  extractTriggerWordsFromMetadata,
  findDuplicateUploadedLoraPath,
  getUploadedLoraRoot,
  mergeTriggerWords,
  nextAvailableLoraPath,
  readSafetensorsMetadataStrict,
  splitTriggerWords,
  writeUploadedLoraSidecar,
} from '@/server/loraLibrary';

export const runtime = 'nodejs';

const MAX_REQUEST_BYTES = 5 * 1024 * 1024 * 1024;
const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;

export async function POST(request: NextRequest) {
  let savedPath: string | null = null;
  let createdPath: string | null = null;

  try {
    const contentLength = Number(request.headers.get('content-length') || 0);
    if (contentLength > MAX_REQUEST_BYTES) {
      return NextResponse.json({ error: 'Request body too large' }, { status: 413 });
    }

    const formData = await request.formData();
    const file = formData.get('file');
    if (!(file instanceof File)) {
      return NextResponse.json({ error: 'No LoRA file provided' }, { status: 400 });
    }

    if (!file.name.toLowerCase().endsWith('.safetensors')) {
      return NextResponse.json({ error: 'LoRA upload must be a .safetensors file' }, { status: 400 });
    }

    if (file.size > MAX_FILE_BYTES) {
      return NextResponse.json({ error: 'LoRA file too large' }, { status: 413 });
    }

    const root = await getUploadedLoraRoot();
    await fs.promises.mkdir(root, { recursive: true });
    const content = Buffer.from(await file.arrayBuffer());
    const duplicatePath = await findDuplicateUploadedLoraPath(root, file.name, content);
    savedPath = duplicatePath || (await nextAvailableLoraPath(root, file.name));
    const reused = duplicatePath !== null;
    if (!reused) {
      createdPath = savedPath;
      await fs.promises.writeFile(savedPath, content);
    }

    const metadata = await readSafetensorsMetadataStrict(savedPath);
    const userTriggerWords = splitTriggerWords(formData.get('trigger_words'));
    const metadataTriggerWords = extractTriggerWordsFromMetadata(metadata);
    const triggerWords = userTriggerWords.length > 0 ? userTriggerWords : metadataTriggerWords;

    if (!reused || userTriggerWords.length > 0) {
      await writeUploadedLoraSidecar(savedPath, {
        originalFilename: file.name,
        uploadedAt: new Date().toISOString(),
        triggerWords: mergeTriggerWords(triggerWords),
        triggerWordSource: userTriggerWords.length > 0 ? 'user' : metadataTriggerWords.length > 0 ? 'metadata' : 'none',
      });
    }

    const lora = await buildUploadedLoraEntry(savedPath);
    return NextResponse.json({ lora, reused });
  } catch (error) {
    if (createdPath) {
      await fs.promises.unlink(createdPath).catch(() => {});
    }

    console.error('LoRA upload error:', error);
    const message = error instanceof Error ? error.message : 'Error uploading LoRA';
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};
