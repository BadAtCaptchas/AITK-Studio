import { execFile } from 'child_process';
import { promisify } from 'util';
import { getToolkitPythonPath } from './pythonPath';
import { TOOLKIT_ROOT } from '../paths';

const execute = promisify(execFile);
const cache = new Map<string, Buffer>();
const pending = new Map<string, Promise<Buffer>>();
/** In-memory previews: no writes into datasets or plaintext encrypted media. */
export async function waveformArtwork(canonicalPath: string, mtimeMs: number, size: number): Promise<Buffer> {
  const key = `${canonicalPath}:${mtimeMs}:${size}`;
  const previous = cache.get(key);
  if (previous) return previous;
  const running = pending.get(key);
  if (running) return running;
  if (pending.size >= 4) throw new Error('Artwork renderer busy');
  const task = execute(
    getToolkitPythonPath(),
    [
      '-c',
      'import sys; from toolkit.audio.album_artwork import create_artwork,load_waveform; create_artwork(load_waveform(sys.argv[1]),size=300).save(sys.stdout.buffer,format="PNG")',
      canonicalPath,
    ],
    { cwd: TOOLKIT_ROOT, windowsHide: true, encoding: 'buffer', timeout: 60000, maxBuffer: 4 * 1024 * 1024 },
  )
    .then(result => {
      if (cache.size >= 128) cache.delete(cache.keys().next().value!);
      cache.set(key, result.stdout);
      return result.stdout;
    })
    .finally(() => pending.delete(key));
  pending.set(key, task);
  return task;
}
