import fs from 'fs';
import path from 'path';
import { MAX_LAYERED_MANIFEST_BYTES, parseLayeredImageManifest } from '../domain/layeredImages';

const pendingName = /^\.pending-([A-Za-z0-9_-]{8,128})\.json$/;

function ownedPaths(root: string, filename: string, text: string): string[] {
  const manifest = parseLayeredImageManifest(JSON.parse(text));
  if (manifest.id !== pendingName.exec(filename)?.[1]) return [];
  return [manifest.composite, manifest.caption].map(asset => path.resolve(root, asset));
}

// A pending manifest is renamed into its group as the final commit. Read it
// after enumerating dataset files, so even a newly published composite stays
// hidden until its complete group becomes discoverable.
export function pendingLayeredImagePaths(root: string): Set<string> {
  const result = new Set<string>();
  const directory = path.join(root, '.layers');
  try {
    if (fs.lstatSync(directory).isSymbolicLink()) return result;
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isFile() || !pendingName.test(entry.name)) continue;
      try {
        const filename = path.join(directory, entry.name);
        if (fs.statSync(filename).size > MAX_LAYERED_MANIFEST_BYTES) continue;
        for (const asset of ownedPaths(root, entry.name, fs.readFileSync(filename, 'utf8'))) result.add(asset);
      } catch {
        // The marker may be in the process of being written or committed.
      }
    }
  } catch {
    // Ordinary datasets do not have a layer directory.
  }
  return result;
}

export async function pendingLayeredImagePathsAsync(root: string): Promise<Set<string>> {
  const result = new Set<string>();
  const directory = path.join(root, '.layers');
  try {
    if ((await fs.promises.lstat(directory)).isSymbolicLink()) return result;
    const entries = await fs.promises.readdir(directory, { withFileTypes: true });
    await Promise.all(entries.map(async entry => {
      if (!entry.isFile() || !pendingName.test(entry.name)) return;
      try {
        const filename = path.join(directory, entry.name);
        if ((await fs.promises.stat(filename)).size > MAX_LAYERED_MANIFEST_BYTES) return;
        for (const asset of ownedPaths(root, entry.name, await fs.promises.readFile(filename, 'utf8'))) result.add(asset);
      } catch {
        // The marker may be in the process of being written or committed.
      }
    }));
  } catch {
    // Ordinary datasets do not have a layer directory.
  }
  return result;
}
