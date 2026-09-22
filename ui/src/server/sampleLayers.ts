import fs from 'fs/promises';
import path from 'path';
import { safeLayeredFile } from './layeredImages';

export type SampleLayers = { root: string; sidecar: string; layers: string[]; paths: string[] };

export async function readSampleLayers(compositePath: string): Promise<SampleLayers | null> {
  const requestedRoot = path.dirname(compositePath);
  const name = path.basename(compositePath) + '.layers.json';
  const stat = await fs.lstat(path.join(requestedRoot, name)).catch(() => null);
  if (!stat) return null;
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 64 * 1024)
    throw new Error('Invalid sample layer manifest');
  const root = await fs.realpath(requestedRoot);
  const sidecar = await safeLayeredFile(root, name);
  const value: unknown = JSON.parse(await fs.readFile(sidecar, 'utf8'));
  if (
    !value ||
    typeof value !== 'object' ||
    !('version' in value) ||
    value.version !== 1 ||
    !('order' in value) ||
    value.order !== 'bottom-to-top' ||
    !('layers' in value) ||
    !Array.isArray(value.layers) ||
    value.layers.length < 1 ||
    value.layers.length > 32
  )
    throw new Error('Invalid sample layer manifest');
  const layers: string[] = [];
  for (const item of value.layers) {
    if (
      typeof item !== 'string' ||
      !/^\.layers\/[A-Za-z0-9_-]{8,128}\/[A-Za-z0-9_-]+\.png$/.test(item) ||
      layers.some(existing => existing.toLowerCase() === item.toLowerCase()) ||
      (layers.length > 0 && item.split('/')[1] !== layers[0].split('/')[1])
    )
      throw new Error('Invalid sample layer path');
    layers.push(item);
  }
  return { root, sidecar, layers, paths: await Promise.all(layers.map(layer => safeLayeredFile(root, layer))) };
}

export async function deleteSampleLayers(compositePath: string): Promise<void> {
  const group = await readSampleLayers(compositePath);
  if (!group) return;
  // A shared or tampered sidecar must not let deletion remove another sample's assets.
  const groupDirectory = path.dirname(group.paths[0]);
  for (const entry of await fs.readdir(group.root)) {
    if (!entry.toLowerCase().endsWith('.layers.json') || path.relative(path.join(group.root, entry), group.sidecar) === '') continue;
    const other = await readSampleLayers(path.join(group.root, entry.slice(0, -'.layers.json'.length)));
    if (other?.paths.some(layer => path.relative(path.dirname(layer), groupDirectory) === ''))
      throw new Error('Sample layers are shared by another composite');
  }
  for (const asset of group.paths) await fs.unlink(asset);
  await fs.unlink(group.sidecar);
  await fs.rmdir(path.dirname(group.paths[0])).catch(() => undefined);
}
