export type LayeredImageManifest = {
  format: 'aitk.layered-image';
  version: 1;
  id: string;
  composite: string;
  caption: string;
  width: number;
  height: number;
  order: 'bottom-to-top';
  layers: Array<{ path: string; name: string; caption: string }>;
  source?: { format: 'psd' | 'ora'; filename: string; sha256: string };
};
export const MAX_LAYERED_MANIFEST_BYTES = 1024 * 1024;

export function isLayeredImageAssetPath(filePath: string): boolean {
  return filePath.split(/[\\/]/).some(segment => segment.toLowerCase() === '.layers');
}

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const safeRelativePath = (value: unknown): value is string =>
  typeof value === 'string' &&
  value.length > 0 &&
  value.length < 1024 &&
  !/[\\:\x00-\x1f]/.test(value) &&
  !value.startsWith('/') &&
  value.split('/').every(part => part !== '' && part !== '.' && part !== '..');

export function parseLayeredImageManifest(value: unknown): LayeredImageManifest {
  if (
    !record(value) ||
    value.format !== 'aitk.layered-image' ||
    value.version !== 1 ||
    typeof value.id !== 'string' ||
    !/^[A-Za-z0-9_-]{8,128}$/.test(value.id) ||
    value.order !== 'bottom-to-top' ||
    !safeRelativePath(value.composite) ||
    value.composite.includes('/') ||
    value.composite.startsWith('.') ||
    !value.composite.endsWith('.png') ||
    !safeRelativePath(value.caption) ||
    value.caption.includes('/') ||
    value.caption !== value.composite.slice(0, -4) + '.txt' ||
    !Number.isSafeInteger(value.width) ||
    !Number.isSafeInteger(value.height) ||
    Number(value.width) < 1 ||
    Number(value.height) < 1 ||
    Number(value.width) > 8192 ||
    Number(value.height) > 8192 ||
    Number(value.width) * Number(value.height) > 32_000_000 ||
    !Array.isArray(value.layers) ||
    value.layers.length < 1 ||
    value.layers.length > 32
  ) {
    throw new Error('Invalid layered image manifest');
  }
  const id = value.id;
  let source: LayeredImageManifest['source'];
  if (value.source !== undefined) {
    const candidate = value.source;
    if (
      !record(candidate) ||
      (candidate.format !== 'psd' && candidate.format !== 'ora') ||
      typeof candidate.filename !== 'string' ||
      candidate.filename.length > 512 ||
      typeof candidate.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(candidate.sha256)
    )
      throw new Error('Invalid layered image source');
    source = { format: candidate.format, filename: candidate.filename, sha256: candidate.sha256 };
  }
  const paths = new Set<string>();
  const layers = value.layers.map((layer: unknown) => {
    if (
      !record(layer) ||
      !safeRelativePath(layer.path) ||
      !layer.path.startsWith(`.layers/${id}/`) ||
      layer.path.split('/').length !== 3 ||
      !layer.path.endsWith('.png') ||
      paths.has(layer.path.toLowerCase()) ||
      typeof layer.name !== 'string' ||
      layer.name.length > 512 ||
      (layer.caption !== undefined && (typeof layer.caption !== 'string' || layer.caption.length > 65_536))
    ) {
      throw new Error('Invalid layered image target');
    }
    paths.add(layer.path.toLowerCase());
    return { path: layer.path, name: layer.name, caption: typeof layer.caption === 'string' ? layer.caption : '' };
  });
  const manifest: LayeredImageManifest = {
    format: 'aitk.layered-image',
    version: 1,
    id,
    composite: value.composite,
    caption: value.caption,
    width: Number(value.width),
    height: Number(value.height),
    order: 'bottom-to-top',
    layers,
    ...(source ? { source } : {}),
  };
  if (new TextEncoder().encode(JSON.stringify(manifest)).length > MAX_LAYERED_MANIFEST_BYTES)
    throw new Error('Layer manifest is too large');
  return manifest;
}

export type LayeredImageDocument = {
  manifest: LayeredImageManifest;
  captionText: string;
  revision: string;
  compositeUrl: string;
  layerUrls: string[];
};

export function parseLayeredDocumentsResponse(value: unknown): LayeredImageDocument[] {
  if (!record(value) || !Array.isArray(value.documents)) throw new Error('Invalid layered document response');
  return value.documents.map((document: unknown) => {
    if (
      !record(document) ||
      typeof document.captionText !== 'string' ||
      typeof document.revision !== 'string' ||
      typeof document.compositeUrl !== 'string' ||
      !Array.isArray(document.layerUrls) ||
      !document.layerUrls.every((url: unknown) => typeof url === 'string' && url.startsWith('/api/')) ||
      !document.compositeUrl.startsWith('/api/')
    )
      throw new Error('Invalid layered document assets');
    const manifest = parseLayeredImageManifest(document.manifest);
    if (document.layerUrls.length !== manifest.layers.length) throw new Error('Layer asset count mismatch');
    return {
      manifest,
      captionText: document.captionText,
      revision: document.revision,
      compositeUrl: document.compositeUrl,
      layerUrls: document.layerUrls,
    };
  });
}
