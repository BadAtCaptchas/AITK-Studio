import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { webcrypto } from 'crypto';
import { getDatasetsRoot } from '@/server/settings';
import {
  getRandomPromptCaptionExtCandidates,
  normalizeRandomPromptCaptionExt,
  parseRandomPromptCaptionText,
  parseRandomPromptCaptionTextAuto,
} from '@/server/randomPromptCaptions';
import {
  getKeyForRequiredDataset,
  isEncryptedDatasetFolder,
  normalizeEncryptedKeyMap,
  readEncryptedManifest,
  resolveEncryptedObjectPath,
} from '@/server/encryptedDatasets';
import type { EncryptedDatasetCatalog, EncryptedDatasetItem, EncryptedDatasetStartKey } from '@/types';

type DatasetPromptRequest = {
  folderPath?: string;
  captionExt?: string;
  defaultCaption?: string;
};

type RandomPromptCandidate = {
  prompt: string;
  datasetPath: string;
  mediaPath: string;
  captionPath?: string;
  source: 'caption' | 'default_caption';
};

type RandomPromptState = {
  selected: RandomPromptCandidate | null;
  candidateCount: number;
  scannedMediaCount: number;
  skippedDatasets: string[];
};

type EncryptedPayload = {
  nonce: string;
  data: string;
};

const mediaExtensions = [
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
];

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function resolveDatasetPath(folderPath: unknown, datasetsRoot: string) {
  if (typeof folderPath !== 'string' || !folderPath.trim()) return null;
  const resolved = path.isAbsolute(folderPath) ? path.resolve(folderPath) : path.resolve(datasetsRoot, folderPath);
  return isPathInside(datasetsRoot, resolved) ? resolved : null;
}

function captionPathForMedia(mediaPath: string, captionExt: string) {
  const parsed = path.parse(mediaPath);
  return path.join(parsed.dir, `${parsed.name}${captionExt}`);
}

function findCaptionForMedia(mediaPath: string, captionExt: string) {
  for (const candidateExt of getRandomPromptCaptionExtCandidates(captionExt)) {
    const candidatePath = captionPathForMedia(mediaPath, candidateExt);
    if (fs.existsSync(candidatePath)) {
      return { path: candidatePath, ext: candidateExt };
    }
  }
  return null;
}

function considerCandidate(state: RandomPromptState, candidate: RandomPromptCandidate) {
  state.candidateCount += 1;
  if (Math.random() < 1 / state.candidateCount) {
    state.selected = candidate;
  }
}

function base64ToBytes(value: string) {
  return Buffer.from(value, 'base64');
}

async function decryptAesGcmToText(keyB64: string, payload: EncryptedPayload, aad: string) {
  const cryptoKey = await webcrypto.subtle.importKey(
    'raw',
    base64ToBytes(keyB64),
    { name: 'AES-GCM', length: 256 },
    false,
    ['decrypt'],
  );
  const decrypted = await webcrypto.subtle.decrypt(
    {
      name: 'AES-GCM',
      iv: base64ToBytes(payload.nonce),
      additionalData: new TextEncoder().encode(aad),
      tagLength: 128,
    },
    cryptoKey,
    base64ToBytes(payload.data),
  );
  return new TextDecoder().decode(decrypted);
}

async function decryptEncryptedCatalog(datasetPath: string, keyB64: string) {
  const manifest = await readEncryptedManifest(datasetPath);
  const catalogText = await decryptAesGcmToText(keyB64, manifest.catalog, 'aitk-encrypted-catalog:v1');
  return JSON.parse(catalogText) as EncryptedDatasetCatalog;
}

async function decryptEncryptedCaption(datasetPath: string, item: EncryptedDatasetItem, keyB64: string) {
  if (!item.captionObjectPath) return '';
  const captionPath = resolveEncryptedObjectPath(datasetPath, item.captionObjectPath);
  if (!fs.existsSync(captionPath)) return '';
  const payload = JSON.parse(fs.readFileSync(captionPath, 'utf-8')) as EncryptedPayload;
  return decryptAesGcmToText(keyB64, payload, `aitk-encrypted-object:${item.captionObjectPath}`);
}

async function scanEncryptedDatasetFolder(
  datasetPath: string,
  keyB64: string,
  defaultCaption: string,
  state: RandomPromptState,
) {
  const catalog = await decryptEncryptedCatalog(datasetPath, keyB64);

  for (const item of catalog.items) {
    state.scannedMediaCount += 1;
    let prompt = '';
    let source: RandomPromptCandidate['source'] = 'caption';

    if (item.captionObjectPath) {
      prompt = parseRandomPromptCaptionTextAuto(await decryptEncryptedCaption(datasetPath, item, keyB64));
    }

    if (!prompt && defaultCaption) {
      prompt = defaultCaption;
      source = 'default_caption';
    }

    if (!prompt) continue;

    considerCandidate(state, {
      prompt,
      datasetPath,
      mediaPath: item.objectPath,
      captionPath: item.captionObjectPath,
      source,
    });
  }
}

function scanDatasetFolder(
  dir: string,
  datasetPath: string,
  captionExt: string,
  defaultCaption: string,
  state: RandomPromptState,
) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;

    const itemPath = path.join(dir, name);
    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      scanDatasetFolder(itemPath, datasetPath, captionExt, defaultCaption, state);
      continue;
    }

    if (!entry.isFile()) continue;
    if (!mediaExtensions.includes(path.extname(name).toLowerCase())) continue;

    state.scannedMediaCount += 1;
    const caption = findCaptionForMedia(itemPath, captionExt);
    let prompt = '';
    let source: RandomPromptCandidate['source'] = 'caption';
    let selectedCaptionPath: string | undefined = caption?.path;

    if (caption) {
      prompt = parseRandomPromptCaptionText(fs.readFileSync(caption.path, 'utf-8'), caption.ext);
    }

    if (!prompt && defaultCaption) {
      prompt = defaultCaption;
      source = 'default_caption';
      selectedCaptionPath = undefined;
    }

    if (!prompt) continue;

    considerCandidate(state, {
      prompt,
      datasetPath,
      mediaPath: itemPath,
      captionPath: selectedCaptionPath,
      source,
    });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasets = Array.isArray(body?.datasets) ? (body.datasets as DatasetPromptRequest[]) : [];
    const encryptedDatasetKeys = Array.isArray(body?.encryptedDatasetKeys)
      ? (body.encryptedDatasetKeys as EncryptedDatasetStartKey[])
      : [];

    if (datasets.length === 0) {
      return NextResponse.json({ error: 'No datasets were provided.' }, { status: 400 });
    }

    const datasetsRoot = path.resolve(await getDatasetsRoot());
    const encryptedKeyMap = normalizeEncryptedKeyMap(encryptedDatasetKeys);
    const state: RandomPromptState = {
      selected: null,
      candidateCount: 0,
      scannedMediaCount: 0,
      skippedDatasets: [],
    };

    for (const dataset of datasets) {
      const datasetPath = resolveDatasetPath(dataset.folderPath, datasetsRoot);
      if (!datasetPath || !fs.existsSync(datasetPath) || !fs.statSync(datasetPath).isDirectory()) {
        state.skippedDatasets.push(String(dataset.folderPath || ''));
        continue;
      }

      if (isEncryptedDatasetFolder(datasetPath)) {
        const keyB64 = getKeyForRequiredDataset(encryptedKeyMap, {
          path: datasetPath,
          name: path.basename(datasetPath),
        });
        if (!keyB64) {
          state.skippedDatasets.push(`${datasetPath} (locked encrypted)`);
          continue;
        }
        await scanEncryptedDatasetFolder(
          datasetPath,
          keyB64,
          typeof dataset.defaultCaption === 'string' ? dataset.defaultCaption.trim() : '',
          state,
        );
        continue;
      }

      scanDatasetFolder(
        datasetPath,
        datasetPath,
        normalizeRandomPromptCaptionExt(dataset.captionExt),
        typeof dataset.defaultCaption === 'string' ? dataset.defaultCaption.trim() : '',
        state,
      );
    }

    if (!state.selected) {
      return NextResponse.json(
        {
          error: 'No captions were found in the configured datasets.',
          scannedMediaCount: state.scannedMediaCount,
          skippedDatasets: state.skippedDatasets,
        },
        { status: 404 },
      );
    }

    return NextResponse.json({
      prompt: state.selected.prompt,
      source: state.selected.source,
      datasetPath: state.selected.datasetPath,
      mediaPath: state.selected.mediaPath,
      captionPath: state.selected.captionPath,
      candidateCount: state.candidateCount,
      scannedMediaCount: state.scannedMediaCount,
      skippedDatasets: state.skippedDatasets,
    });
  } catch (error) {
    console.error('Error importing random dataset prompt:', error);
    return NextResponse.json({ error: 'Failed to import a random dataset prompt.' }, { status: 500 });
  }
}
