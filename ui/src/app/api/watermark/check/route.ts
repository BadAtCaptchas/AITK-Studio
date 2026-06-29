import { spawn } from 'child_process';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import { NextRequest, NextResponse } from 'next/server';
import { TOOLKIT_ROOT } from '@/paths';
import { getToolkitPythonPath } from '@/server/tensorboard';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

const MAX_IMAGE_BYTES = 50 * 1024 * 1024;
const MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024;
const allowedExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl']);

type UploadedFile = {
  name: string;
  size: number;
  arrayBuffer: () => Promise<ArrayBuffer>;
};

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function isUploadedFile(value: FormDataEntryValue | null): value is UploadedFile {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as UploadedFile).name === 'string' &&
    typeof (value as UploadedFile).size === 'number' &&
    typeof (value as UploadedFile).arrayBuffer === 'function'
  );
}

function positiveInteger(value: FormDataEntryValue | null, fallback: number) {
  const parsed = Number(value ?? fallback);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function normalizedThreshold(value: FormDataEntryValue | null) {
  const parsed = Number(value ?? 0.75);
  if (!Number.isFinite(parsed)) return 0.75;
  return Math.max(0, Math.min(1, parsed));
}

function runChecker(args: string[]) {
  return new Promise<Record<string, unknown>>((resolve, reject) => {
    const child = spawn(getToolkitPythonPath(), args, {
      cwd: TOOLKIT_ROOT,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
      },
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill();
      reject(new Error('Watermark check timed out.'));
    }, 120000);

    const appendOutput = (current: string, chunk: Buffer) => {
      const next = current + chunk.toString('utf8');
      return next.length > MAX_PROCESS_OUTPUT_BYTES ? next.slice(-MAX_PROCESS_OUTPUT_BYTES) : next;
    };

    child.stdout.on('data', chunk => {
      stdout = appendOutput(stdout, chunk);
    });
    child.stderr.on('data', chunk => {
      stderr = appendOutput(stderr, chunk);
    });
    child.once('error', error => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(error);
    });
    child.once('exit', code => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (code !== 0) {
        try {
          const parsed = JSON.parse(stderr.trim().split(/\r?\n/).pop() || '{}');
          if (typeof parsed?.error === 'string') {
            reject(new Error(parsed.error));
            return;
          }
        } catch {
          // Fall through to the generic message below.
        }
        reject(new Error(stderr.trim() || `Watermark checker exited with code ${code ?? 'unknown'}.`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error('Watermark checker returned invalid JSON.'));
      }
    });
  });
}

export async function POST(request: NextRequest) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  let tempDir: string | null = null;
  try {
    const contentLength = Number(request.headers.get('content-length') || 0);
    if (contentLength > MAX_IMAGE_BYTES + 4096) {
      return NextResponse.json({ error: 'Image upload is too large.' }, { status: 413 });
    }

    const formData = await request.formData();
    const file = formData.get('image');
    if (!isUploadedFile(file)) {
      return NextResponse.json({ error: 'Image file is required.' }, { status: 400 });
    }
    if (file.size > MAX_IMAGE_BYTES) {
      return NextResponse.json({ error: 'Image upload is too large.' }, { status: 413 });
    }

    const codec = String(formData.get('codec') || 'builtin:authenlora_48bits').trim();
    const msgBits = positiveInteger(formData.get('msg_bits'), 48);
    const threshold = normalizedThreshold(formData.get('threshold'));
    const expectedSecret = String(formData.get('expected_secret') || '').trim();
    if (expectedSecret && (expectedSecret.length !== msgBits || /[^01]/.test(expectedSecret))) {
      return NextResponse.json({ error: 'Expected secret must be binary and match message bits.' }, { status: 400 });
    }
    if (!codec) {
      return NextResponse.json({ error: 'Codec is required.' }, { status: 400 });
    }

    const extension = path.extname(file.name || '').toLowerCase() || '.png';
    if (!allowedExtensions.has(extension)) {
      return NextResponse.json({ error: 'Unsupported image type.' }, { status: 400 });
    }

    tempDir = path.join(TOOLKIT_ROOT, '.tmp', 'watermark-check', randomUUID());
    await fsp.mkdir(tempDir, { recursive: true });
    const imagePath = path.join(tempDir, `image${extension}`);
    await fsp.writeFile(imagePath, Buffer.from(await file.arrayBuffer()));

    const args = [
      path.join('scripts', 'check_authenlora_watermark.py'),
      '--image',
      imagePath,
      '--codec',
      codec,
      '--msg-bits',
      String(msgBits),
      '--threshold',
      String(threshold),
    ];
    if (expectedSecret) {
      args.push('--expected-secret', expectedSecret);
    }

    const result = await runChecker(args);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Watermark check error:', error);
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Watermark check failed.' }, { status: 500 });
  } finally {
    if (tempDir) {
      try {
        await fsp.rm(tempDir, { recursive: true, force: true });
      } catch {
        if (fs.existsSync(tempDir)) {
          console.warn(`Unable to remove watermark check temp dir: ${tempDir}`);
        }
      }
    }
  }
}
