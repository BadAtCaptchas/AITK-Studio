import fs from 'fs';
import path from 'path';
import { getDataRoot } from './settings';

const UPLOADED_LORA_DIR = 'loras';
const SIDECAR_SUFFIX = '.aitk.json';
const MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024;

export type LoraTriggerWordSource = 'metadata' | 'user' | 'none';

export type LoraLibraryEntry = {
  id: string;
  label: string;
  path: string;
  filename: string;
  source: 'job' | 'uploaded';
  updatedAt: string;
  sizeBytes: number;
  triggerWords: string[];
  triggerWordSource: LoraTriggerWordSource;
  originalFilename?: string;
};

type UploadedLoraSidecar = {
  originalFilename?: string;
  uploadedAt?: string;
  triggerWords?: string[];
  triggerWordSource?: LoraTriggerWordSource;
};

type SafetensorsMetadata = Record<string, string>;

function normalizeTriggerWord(value: string) {
  const normalized = value
    .trim()
    .replace(/^["'`]+|["'`]+$/g, '')
    .replace(/\s+/g, ' ');
  if (!normalized || ['none', 'null', 'undefined', 'n/a'].includes(normalized.toLowerCase())) {
    return null;
  }
  return normalized.slice(0, 200);
}

export function mergeTriggerWords(...lists: unknown[]) {
  const seen = new Set<string>();
  const merged: string[] = [];

  for (const list of lists) {
    for (const word of splitTriggerWords(list)) {
      const key = word.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(word);
      if (merged.length >= 24) return merged;
    }
  }

  return merged;
}

export function splitTriggerWords(value: unknown): string[] {
  if (value == null) return [];
  if (Array.isArray(value)) {
    return mergeTriggerWords(...value);
  }
  if (typeof value !== 'string') {
    return [];
  }

  const trimmed = value.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed !== trimmed) {
      return splitTriggerWords(parsed);
    }
  } catch {}

  return trimmed
    .split(/[,\n;]/)
    .map(normalizeTriggerWord)
    .filter((word): word is string => Boolean(word));
}

function parseJsonString(value: string) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function collectNestedTriggerWords(value: unknown, depth = 0): string[] {
  if (depth > 4 || value == null) return [];
  if (typeof value === 'string' || Array.isArray(value)) return splitTriggerWords(value);
  if (typeof value !== 'object') return [];

  const triggerWords: string[] = [];
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    if (
      [
        'trigger',
        'trigger_word',
        'trigger_words',
        'trained_word',
        'trained_words',
        'trainedwords',
        'activation_text',
        'activation_tags',
        'trigger_phrase',
      ].includes(normalizedKey)
    ) {
      triggerWords.push(...splitTriggerWords(child));
    } else if (typeof child === 'object' && child !== null) {
      triggerWords.push(...collectNestedTriggerWords(child, depth + 1));
    }
  }
  return mergeTriggerWords(triggerWords);
}

function triggerWordsFromDatasetDirs(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];

  const words: string[] = [];
  for (const [dirName, tagMap] of Object.entries(value as Record<string, unknown>)) {
    const candidate = normalizeTriggerWord(dirName.replace(/^\d+[_-]?/, ''));
    if (!candidate) continue;

    if (tagMap && typeof tagMap === 'object' && !Array.isArray(tagMap)) {
      const tags = Object.keys(tagMap);
      if (tags.length === 0 || tags.some(tag => tag.toLowerCase() === candidate.toLowerCase())) {
        words.push(candidate);
      }
    } else {
      words.push(candidate);
    }
  }
  return mergeTriggerWords(words);
}

export function extractTriggerWordsFromMetadata(metadata: SafetensorsMetadata) {
  const directKeys = new Set([
    'trigger',
    'trigger_word',
    'trigger_words',
    'trained_word',
    'trained_words',
    'trainedwords',
    'activation_text',
    'activation_tags',
    'modelspec_trigger_phrase',
    'ss_trigger_word',
    'aitk_trigger_word',
    'civitai_trained_words',
  ]);
  const datasetKeys = new Set(['ss_tag_frequency', 'ss_dataset_dirs']);
  const nestedKeys = new Set(['training_info', 'modelspec', 'civitai']);
  const words: string[] = [];

  for (const [key, value] of Object.entries(metadata || {})) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    if (directKeys.has(normalizedKey)) {
      words.push(...splitTriggerWords(value));
      continue;
    }

    if (datasetKeys.has(normalizedKey)) {
      words.push(...triggerWordsFromDatasetDirs(parseJsonString(value)));
      continue;
    }

    if (nestedKeys.has(normalizedKey)) {
      const parsed = parseJsonString(value);
      words.push(...collectNestedTriggerWords(parsed));
    }
  }

  return mergeTriggerWords(words);
}

async function readSafetensorsHeader(filePath: string) {
  const handle = await fs.promises.open(filePath, 'r');
  try {
    const stat = await handle.stat();
    if (stat.size < 9) {
      throw new Error('Safetensors file is too small.');
    }

    const prefix = Buffer.alloc(8);
    await handle.read(prefix, 0, 8, 0);
    const headerLengthBig = prefix.readBigUInt64LE(0);
    if (headerLengthBig > BigInt(MAX_SAFETENSORS_HEADER_BYTES)) {
      throw new Error('Safetensors metadata header is too large.');
    }

    const headerLength = Number(headerLengthBig);
    if (!Number.isFinite(headerLength) || headerLength <= 0 || headerLength > stat.size - 8) {
      throw new Error('Safetensors metadata header is invalid.');
    }

    const headerBuffer = Buffer.alloc(headerLength);
    await handle.read(headerBuffer, 0, headerLength, 8);
    return JSON.parse(headerBuffer.toString('utf8'));
  } finally {
    await handle.close();
  }
}

export async function readSafetensorsMetadata(filePath: string) {
  try {
    const header = await readSafetensorsHeader(filePath);
    const metadata = header?.__metadata__;
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(metadata).map(([key, value]) => [
        key,
        typeof value === 'string' ? value : JSON.stringify(value),
      ]),
    ) as SafetensorsMetadata;
  } catch {
    return {};
  }
}

export async function readSafetensorsMetadataStrict(filePath: string) {
  const header = await readSafetensorsHeader(filePath);
  const metadata = header?.__metadata__;
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(metadata).map(([key, value]) => [key, typeof value === 'string' ? value : JSON.stringify(value)]),
  ) as SafetensorsMetadata;
}

export async function getUploadedLoraRoot() {
  return path.join(await getDataRoot(), UPLOADED_LORA_DIR);
}

export function sanitizeLoraFilename(filename: string) {
  const rawName = path.basename(filename.replace(/\\/g, '/'));
  const ext = path.extname(rawName).toLowerCase();
  const stem = path
    .basename(rawName, ext)
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 120);
  return `${stem || 'uploaded_lora'}.safetensors`;
}

export async function nextAvailableLoraPath(root: string, filename: string) {
  const safeFilename = sanitizeLoraFilename(filename);
  const ext = path.extname(safeFilename);
  const stem = path.basename(safeFilename, ext);
  let candidate = path.join(root, safeFilename);
  let index = 1;
  while (fs.existsSync(candidate)) {
    candidate = path.join(root, `${stem}_${index}${ext}`);
    index += 1;
  }
  return candidate;
}

function sidecarPathFor(filePath: string) {
  return `${filePath}${SIDECAR_SUFFIX}`;
}

async function readSidecar(filePath: string): Promise<UploadedLoraSidecar> {
  try {
    return JSON.parse(await fs.promises.readFile(sidecarPathFor(filePath), 'utf8'));
  } catch {
    return {};
  }
}

export async function writeUploadedLoraSidecar(filePath: string, sidecar: UploadedLoraSidecar) {
  await fs.promises.writeFile(sidecarPathFor(filePath), JSON.stringify(sidecar, null, 2), 'utf8');
}

export async function buildUploadedLoraEntry(filePath: string): Promise<LoraLibraryEntry | null> {
  const stat = await fs.promises.stat(filePath).catch(() => null);
  if (!stat?.isFile()) return null;

  const sidecar = await readSidecar(filePath);
  const metadata = await readSafetensorsMetadata(filePath);
  const metadataTriggerWords = extractTriggerWordsFromMetadata(metadata);
  const sidecarTriggerWords = splitTriggerWords(sidecar.triggerWords);
  const triggerWords = sidecarTriggerWords.length > 0 ? sidecarTriggerWords : metadataTriggerWords;
  const triggerWordSource =
    sidecarTriggerWords.length > 0
      ? sidecar.triggerWordSource || 'user'
      : metadataTriggerWords.length > 0
        ? 'metadata'
        : 'none';
  const filename = path.basename(filePath);
  const originalFilename = sidecar.originalFilename || filename;

  return {
    id: `uploaded:${filename}`,
    label: `Uploaded / ${originalFilename}`,
    path: filePath,
    filename,
    source: 'uploaded',
    updatedAt: stat.mtime.toISOString(),
    sizeBytes: stat.size,
    originalFilename,
    triggerWords,
    triggerWordSource,
  };
}

export async function listUploadedLoras() {
  const root = await getUploadedLoraRoot();
  const entries = await fs.promises.readdir(root, { withFileTypes: true }).catch(() => []);
  const loras: LoraLibraryEntry[] = [];

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.safetensors')) continue;
    const lora = await buildUploadedLoraEntry(path.join(root, entry.name));
    if (lora) loras.push(lora);
  }

  return loras;
}
