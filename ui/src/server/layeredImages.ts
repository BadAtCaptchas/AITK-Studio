import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { createHash, randomUUID } from 'crypto';
import {
  MAX_LAYERED_MANIFEST_BYTES,
  parseLayeredImageManifest,
  type LayeredImageManifest,
} from '../domain/layeredImages';
import sharp from 'sharp';
import { isPathInside, resolveDatasetDirectoryInsideRoot } from './remoteCaptionSecurity';
import { isEncryptedDatasetFolder } from './encryptedDatasets';

export class LayeredImageError extends Error {
  constructor(
    message: string,
    public status = 400,
  ) {
    super(message);
    this.name = 'LayeredImageError';
  }
}

export async function safeLayeredFile(root: string, relativePath: string): Promise<string> {
  const normalized = relativePath.replace(/\\/g, '/');
  if (
    !normalized ||
    normalized.startsWith('/') ||
    /[:\x00-\x1f]/.test(normalized) ||
    normalized.split('/').some(part => !part || part === '.' || part === '..')
  )
    throw new LayeredImageError('Invalid layer path');
  const canonicalRoot = await fsp.realpath(root);
  let current = canonicalRoot;
  for (const part of normalized.split('/')) {
    current = path.join(current, part);
    const stat = await fsp.lstat(current);
    if (stat.isSymbolicLink()) throw new LayeredImageError('Layer groups cannot contain symbolic links or junctions');
    const canonical = await fsp.realpath(current);
    if (!isPathInside(canonicalRoot, canonical)) throw new LayeredImageError('Layer path escapes dataset');
    current = canonical;
  }
  if (!(await fsp.stat(current)).isFile()) throw new LayeredImageError('Layer asset is not a regular file');
  return current;
}

export async function listLayeredImages(root: string): Promise<LayeredImageManifest[]> {
  const directory = path.join(root, '.layers');
  if (!fs.existsSync(directory)) return [];
  if (isEncryptedDatasetFolder(root))
    throw new LayeredImageError('Layered image groups are not supported in encrypted datasets');
  if ((await fsp.lstat(directory)).isSymbolicLink())
    throw new LayeredImageError('Layer directory cannot be a symbolic link');
  const entries = await fsp.readdir(directory, { withFileTypes: true });
  const groups: LayeredImageManifest[] = [];
  const composites = new Set<string>();
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    if (!entry.isDirectory() || !/^[A-Za-z0-9_-]{8,128}$/.test(entry.name))
      throw new LayeredImageError('Invalid layer group directory');
    // The converter publishes manifest.json last; incomplete imports are not documents yet.
    if (!fs.existsSync(path.join(directory, entry.name, 'manifest.json'))) continue;
    const manifestPath = await safeLayeredFile(root, `.layers/${entry.name}/manifest.json`);
    if ((await fsp.stat(manifestPath)).size > MAX_LAYERED_MANIFEST_BYTES)
      throw new LayeredImageError('Layer manifest is too large');
    const manifest = parseLayeredImageManifest(JSON.parse(await fsp.readFile(manifestPath, 'utf8')));
    if (manifest.id !== entry.name || composites.has(manifest.composite.toLowerCase()))
      throw new LayeredImageError('Duplicate or mismatched layer group');
    composites.add(manifest.composite.toLowerCase());
    for (const asset of [manifest.composite, manifest.caption, ...manifest.layers.map(layer => layer.path)])
      await safeLayeredFile(root, asset);
    groups.push(manifest);
  }
  return groups;
}

export async function validateLayeredImageAssets(root: string, groups?: LayeredImageManifest[]): Promise<void> {
  for (const group of groups || (await listLayeredImages(root))) {
    for (const asset of [group.composite, ...group.layers.map(layer => layer.path)]) {
      const metadata = await sharp(await safeLayeredFile(root, asset), { limitInputPixels: 32_000_000 }).metadata();
      if (
        metadata.format !== 'png' ||
        metadata.channels !== 4 ||
        !metadata.hasAlpha ||
        metadata.width !== group.width ||
        metadata.height !== group.height ||
        (metadata.pages ?? 1) !== 1
      )
        throw new LayeredImageError('Every composite and target must be a full-canvas static RGBA PNG');
    }
  }
}

export async function findLayeredImageForPath(
  filePath: string,
  datasetsRoot: string,
): Promise<{ root: string; manifest: LayeredImageManifest } | null> {
  const relative = path.relative(path.resolve(datasetsRoot), path.resolve(filePath));
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null;
  const requestedRoot = path.join(datasetsRoot, relative.split(path.sep)[0]);
  if (!fs.existsSync(path.join(requestedRoot, '.layers'))) return null;
  const root = await resolveDatasetDirectoryInsideRoot(requestedRoot, datasetsRoot);
  const canonicalFile = await safeLayeredFile(root, path.relative(requestedRoot, path.resolve(filePath)));
  // realpath preserves filesystem identity; path.relative also respects Windows
  // case insensitivity without conflating distinct files on Linux.
  const group = (await listLayeredImages(root)).find(
    manifest =>
      path.relative(path.join(root, manifest.composite), canonicalFile) === '' ||
      path.relative(path.join(root, manifest.caption), canonicalFile) === '' ||
      isPathInside(path.join(root, '.layers', manifest.id), canonicalFile),
  );
  return group ? { root, manifest: group } : null;
}

export async function deleteLayeredImageForPath(filePath: string, datasetsRoot: string): Promise<boolean> {
  const group = await findLayeredImageForPath(filePath, datasetsRoot);
  if (!group) return false;
  const canonicalFile = await fsp.realpath(filePath);
  const composite = await safeLayeredFile(group.root, group.manifest.composite);
  if (path.relative(composite, canonicalFile) !== '') {
    throw new LayeredImageError(
      'Delete a layered document through its composite; individual layer assets cannot be deleted',
    );
  }
  await deleteLayeredImage(group.root, group.manifest);
  return true;
}

export async function deleteLayeredImage(root: string, manifest: LayeredImageManifest): Promise<void> {
  const assets = [
    manifest.composite,
    manifest.caption,
    ...manifest.layers.map(layer => layer.path),
    `.layers/${manifest.id}/manifest.json`,
  ];
  const paths = await Promise.all(assets.map(asset => safeLayeredFile(root, asset)));
  // Only unlink validated manifest-owned files; never recursively delete a user-controlled directory.
  for (const asset of paths) await fsp.unlink(asset);
  await fsp.rmdir(path.join(root, '.layers', manifest.id)).catch(() => undefined);
}

export async function copyLayeredImage(
  root: string,
  manifest: LayeredImageManifest,
  destination: string,
  compositeName: string,
  signal?: AbortSignal,
): Promise<LayeredImageManifest> {
  const checkAbort = () => {
    if (signal?.aborted) throw new LayeredImageError('Layered import cancelled', 499);
  };
  checkAbort();
  if (isEncryptedDatasetFolder(destination))
    throw new LayeredImageError('Layered documents cannot be copied into encrypted datasets');
  const canonicalDestination = await fsp.realpath(destination);
  const layerDirectory = path.join(canonicalDestination, '.layers');
  await fsp.mkdir(layerDirectory, { recursive: true });
  if (
    (await fsp.lstat(layerDirectory)).isSymbolicLink() ||
    !isPathInside(canonicalDestination, await fsp.realpath(layerDirectory))
  )
    throw new LayeredImageError('Invalid destination layer directory');
  const id = randomUUID().replace(/-/g, '');
  const next = parseLayeredImageManifest({
    ...manifest,
    id,
    composite: compositeName,
    caption: compositeName.slice(0, -4) + '.txt',
    layers: manifest.layers.map((layer, index) => ({
      ...layer,
      path: `.layers/${id}/${String(index).padStart(3, '0')}.png`,
    })),
  });
  const targetDir = path.join(layerDirectory, id);
  await fsp.mkdir(targetDir);
  const created: string[] = [];
  try {
    for (const [source, target] of manifest.layers.map((layer, index) => [layer.path, next.layers[index].path])) {
      checkAbort();
      const destinationPath = path.join(canonicalDestination, target);
      await fsp.copyFile(await safeLayeredFile(root, source), destinationPath, fs.constants.COPYFILE_EXCL);
      created.push(destinationPath);
    }
    const stagedAssets = [
      [manifest.caption, path.join(targetDir, '.caption.tmp'), path.join(canonicalDestination, next.caption)],
      [manifest.composite, path.join(targetDir, '.composite.tmp'), path.join(canonicalDestination, next.composite)],
    ];
    for (const [source, temporary] of stagedAssets) {
      checkAbort();
      await fsp.copyFile(await safeLayeredFile(root, source), temporary, fs.constants.COPYFILE_EXCL);
      created.push(temporary);
    }
    // Discovery uses this pending marker to hide root assets until the entire
    // document is committed, including the gap between publishing its files.
    const manifestPath = path.join(targetDir, 'manifest.json');
    const temporaryManifest = path.join(layerDirectory, `.pending-${id}.json`);
    await fsp.writeFile(temporaryManifest, JSON.stringify(next, null, 2), { flag: 'wx' });
    created.push(temporaryManifest);
    for (const [, temporary, target] of stagedAssets) {
      checkAbort();
      try {
        // A hard link publishes the complete file atomically without replacing
        // an existing user's file. Both locations are on the same filesystem.
        await fsp.link(temporary, target);
      } catch (error) {
        const code = error && typeof error === 'object' && 'code' in error ? error.code : undefined;
        if (!['ENOTSUP', 'EOPNOTSUPP', 'EXDEV', 'EPERM', 'ENOSYS'].includes(String(code))) throw error;
        // Filesystems without hard links still remain hidden by the marker.
        await fsp.copyFile(temporary, target, fs.constants.COPYFILE_EXCL);
      }
      created.push(target);
      await fsp.unlink(temporary);
    }
    checkAbort();
    await fsp.rename(temporaryManifest, manifestPath);
    created.push(manifestPath);
    return next;
  } catch (error) {
    // Remove visible assets before the pending marker stops hiding them.
    for (const target of created.reverse()) await fsp.unlink(target).catch(() => undefined);
    await fsp.rmdir(targetDir).catch(() => undefined);
    throw error;
  }
}

export function layeredRevision(manifestText: string, captionText: string): string {
  return createHash('sha256').update(manifestText).update('\0').update(captionText).digest('hex');
}

const pendingSaves = new Map<string, Promise<void>>();
export async function saveLayeredCaptions(
  root: string,
  id: string,
  revision: unknown,
  caption: unknown,
  layers: unknown,
): Promise<void> {
  if (typeof caption !== 'string' || caption.length > 65_536 || typeof revision !== 'string' || !Array.isArray(layers))
    throw new LayeredImageError('Invalid layer captions');
  const key = path.resolve(root, '.layers', id);
  const previous = pendingSaves.get(key) || Promise.resolve();
  const operation = previous
    .catch(() => undefined)
    .then(async () => {
      const manifest = (await listLayeredImages(root)).find(group => group.id === id);
      if (!manifest) throw new LayeredImageError('Layered document not found', 404);
      const manifestPath = await safeLayeredFile(root, `.layers/${id}/manifest.json`);
      const captionPath = await safeLayeredFile(root, manifest.caption);
      const [manifestText, captionText] = await Promise.all([
        fsp.readFile(manifestPath, 'utf8'),
        fsp.readFile(captionPath, 'utf8'),
      ]);
      if (revision !== layeredRevision(manifestText, captionText))
        throw new LayeredImageError('This document changed. Reload before saving.', 409);
      if (layers.length !== manifest.layers.length)
        throw new LayeredImageError('Layer count cannot be changed while editing captions');
      const next = parseLayeredImageManifest({
        ...manifest,
        layers: manifest.layers.map((layer, index) => {
          const edit: unknown = layers[index];
          if (!edit || typeof edit !== 'object' || !('name' in edit) || !('caption' in edit))
            throw new LayeredImageError('Invalid layer caption');
          return { ...layer, name: edit.name, caption: edit.caption };
        }),
      });
      const temporary = `${manifestPath}.${randomUUID()}.tmp`;
      try {
        await fsp.writeFile(temporary, JSON.stringify(next, null, 2), { flag: 'wx' });
        await fsp.writeFile(captionPath, caption);
        try {
          await fsp.rename(temporary, manifestPath);
        } catch (error) {
          await fsp.writeFile(captionPath, captionText);
          throw error;
        }
      } finally {
        await fsp.rm(temporary, { force: true }).catch(() => undefined);
      }
    });
  pendingSaves.set(key, operation);
  try {
    await operation;
  } finally {
    if (pendingSaves.get(key) === operation) pendingSaves.delete(key);
  }
}
