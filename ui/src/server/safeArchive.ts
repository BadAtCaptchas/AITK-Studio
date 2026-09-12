import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Transform } from 'stream';
import { pipeline } from 'stream/promises';
import yauzl from 'yauzl';
import { isPathWithinRoot } from './pathContainment';
import { validateArchiveEntryName } from './trainingJobTransfer';

/** Extraction is confined to an empty staging directory, with actual expansion accounting. */
export async function extractZipSafely(
  zipPath: string,
  destination: string,
  limits: { maxEntries?: number; maxExpandedBytes?: number } = {},
): Promise<void> {
  const maxEntries = limits.maxEntries ?? 100_000,
    maxBytes = limits.maxExpandedBytes ?? 128 * 1024 ** 3;
  await fsp.mkdir(destination, { recursive: true });
  const root = await fsp.realpath(destination);
  if ((await fsp.readdir(root)).length) throw new Error('Archive extraction requires an empty staging directory');
  let entries = 0,
    expanded = 0,
    declared = 0;
  await new Promise<void>((resolve, reject) => {
    yauzl.open(zipPath, { lazyEntries: true, validateEntrySizes: true }, (error, zip) => {
      if (error || !zip) {
        reject(error || new Error('Could not open archive'));
        return;
      }
      let failed = false;
      const fail = (cause: unknown) => {
        if (failed) return;
        failed = true;
        zip.close();
        reject(cause);
      };
      zip.on('error', fail);
      zip.on('end', () => {
        if (!failed) resolve();
      });
      zip.on('entry', entry => {
        void (async () => {
          if (++entries > maxEntries) throw new Error('Archive entry limit exceeded');
          const mode = (entry.externalFileAttributes >>> 16) & 0xf000;
          if (mode !== 0 && mode !== 0x8000 && mode !== 0x4000)
            throw new Error('Archive links and special files are not allowed');
          if (
            !Number.isSafeInteger(entry.uncompressedSize) ||
            entry.uncompressedSize < 0 ||
            (declared += entry.uncompressedSize) > maxBytes
          )
            throw new Error('Archive expansion limit exceeded');
          const relative = validateArchiveEntryName(entry.fileName);
          if (
            relative
              .split('/')
              .filter(Boolean)
              .some(
                segment =>
                  /[<>:"|?*\x00-\x1f]/.test(segment) ||
                  /[ .]$/.test(segment) ||
                  /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(segment),
              )
          )
            throw new Error('Archive path is not portable');
          const target = path.resolve(root, ...relative.split('/'));
          if (!isPathWithinRoot(root, target) || target === root)
            throw new Error('Archive path escapes staging directory');
          if (relative.endsWith('/')) {
            await fsp.mkdir(target, { recursive: true });
            return;
          }
          await fsp.mkdir(path.dirname(target), { recursive: true });
          const parent = await fsp.realpath(path.dirname(target));
          if (!isPathWithinRoot(root, parent)) throw new Error('Archive parent escapes staging directory');
          const space = await fsp.statfs(root);
          if (space.bavail * space.bsize < entry.uncompressedSize + 2 * 1024 ** 3)
            throw new Error('Insufficient disk space for archive extraction');
          const source = await new Promise<NodeJS.ReadableStream>((resolveStream, rejectStream) =>
            zip.openReadStream(entry, (streamError, stream) =>
              streamError || !stream
                ? rejectStream(streamError || new Error('Cannot read archive entry'))
                : resolveStream(stream),
            ),
          );
          const limiter = new Transform({
            transform(chunk: Buffer, _encoding, callback) {
              expanded += chunk.length;
              callback(expanded > maxBytes ? new Error('Archive expansion limit exceeded') : null, chunk);
            },
          });
          await pipeline(source, limiter, fs.createWriteStream(target, { flags: 'wx' }));
        })()
          .then(() => {
            if (!failed) zip.readEntry();
          })
          .catch(fail);
      });
      zip.readEntry();
    });
  });
}
