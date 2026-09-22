import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
import { getToolkitPythonPath } from './pythonPath';
import { copyLayeredImage, LayeredImageError, listLayeredImages, validateLayeredImageAssets } from './layeredImages';

export const MAX_LAYERED_UPLOAD_BYTES = 512 * 1024 * 1024;
export const LAYERED_IMPORT_TIMEOUT_MS = 5 * 60 * 1000;
let activeImports = 0;

export async function runLayeredConverter(
  input: string,
  output: string,
  filename: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return new Promise<unknown>((resolve, reject) => {
    const child = spawn(
      getToolkitPythonPath(),
      ['-u', '-m', 'toolkit.layered_import', '--input', input, '--output', output, '--name', filename],
      {
        cwd: TOOLKIT_ROOT,
        windowsHide: true,
        shell: false,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      },
    );
    let stdout = '',
      stderr = '';
    let stopped: Error | undefined;
    const stop = (error: Error) => {
      stopped = error;
      child.kill('SIGKILL');
    };
    const abort = () => stop(new LayeredImageError('Layered import cancelled', 499));
    const timer = setTimeout(
      () => stop(new LayeredImageError('Layered import exceeded the five minute limit', 408)),
      LAYERED_IMPORT_TIMEOUT_MS,
    );
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
      if (stdout.length > 1024 * 1024) stop(new LayeredImageError('Layered importer output exceeded its limit'));
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr = (stderr + chunk.toString('utf8')).slice(-16_384);
    });
    child.once('error', error => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      reject(error);
    });
    child.once('close', code => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      if (stopped) {
        reject(stopped);
        return;
      }
      if (code !== 0) {
        reject(new LayeredImageError(stderr.trim() || 'Layered document conversion failed'));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new LayeredImageError('Invalid layered importer response', 500));
      }
    });
  });
}

export async function importLayeredDocument(
  input: string,
  output: string,
  filename: string,
  signal?: AbortSignal,
  convert: typeof runLayeredConverter = runLayeredConverter,
) {
  if (signal?.aborted) throw new LayeredImageError('Layered import cancelled', 499);
  if (!/\.(psd|ora)$/i.test(filename) || filename.length > 512 || /[\\/\x00-\x1f]/.test(filename)) {
    throw new LayeredImageError('Select a PSD or OpenRaster (.ora) document');
  }
  if (activeImports >= 1) throw new LayeredImageError('Another layered import is running. Try again shortly.', 429);
  activeImports++;
  let staging: string | undefined;
  try {
    // The subprocess never writes into the user dataset. Forced termination can only leave
    // disposable files inside this exclusively created conversion directory.
    staging = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-layer-convert-'));
    const result = await convert(input, staging, filename, signal);
    if (signal?.aborted) throw new LayeredImageError('Layered import cancelled', 499);
    if (!result || typeof result !== 'object' || !('id' in result) || typeof result.id !== 'string')
      throw new LayeredImageError('Invalid import result', 500);
    const manifest = (await listLayeredImages(staging)).find(group => group.id === result.id);
    if (!manifest) throw new LayeredImageError('Imported layer manifest is missing', 500);
    await validateLayeredImageAssets(staging, [manifest]);
    const warnings =
      'warnings' in result && Array.isArray(result.warnings)
        ? result.warnings.filter((value): value is string => typeof value === 'string')
        : [];
    const published = await copyLayeredImage(staging, manifest, output, manifest.composite, signal);
    return { id: published.id, layer_count: published.layers.length, composite: published.composite, warnings };
  } finally {
    activeImports--;
    if (staging) await fs.rm(staging, { recursive: true, force: true });
  }
}
