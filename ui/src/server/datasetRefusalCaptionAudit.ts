import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { findExistingCaptionSidecar, isTextCaptionFilePath } from './captionFiles';
import { findDatasetItemsRecursively } from './datasetImages';
import { isRefusalCaption } from '../utils/captionQuality';

const AUDIT_VERSION = 'refusal-caption-audit-v1';
const READ_CONCURRENCY = 24;
const MAX_CACHE_ENTRIES = 128;

export type RefusalCaptionAuditResult = {
  datasetFingerprint: string;
  scanned: number;
  refusalCount: number;
  refusals: Record<string, string>;
  cached: boolean;
};

type CaptionAuditCandidate = {
  itemPath: string;
  captionPath: string | null;
  itemSize: number;
  itemMtimeMs: number;
  captionSize: number;
  captionMtimeMs: number;
};

type CachedAudit = Omit<RefusalCaptionAuditResult, 'cached'>;

declare global {
  // eslint-disable-next-line no-var
  var __datasetRefusalCaptionAuditCache: Map<string, CachedAudit> | undefined;
}

const auditCache = globalThis.__datasetRefusalCaptionAuditCache ?? new Map<string, CachedAudit>();

if (!globalThis.__datasetRefusalCaptionAuditCache) {
  globalThis.__datasetRefusalCaptionAuditCache = auditCache;
}

function fileStat(filePath: string) {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

function captionPathForItem(itemPath: string) {
  if (isTextCaptionFilePath(itemPath)) return itemPath;
  return findExistingCaptionSidecar(itemPath);
}

function collectCaptionAuditCandidates(datasetFolder: string): CaptionAuditCandidate[] {
  return findDatasetItemsRecursively(datasetFolder)
    .sort((left, right) => left.localeCompare(right))
    .map(itemPath => {
      const itemStat = fileStat(itemPath);
      const captionPath = captionPathForItem(itemPath);
      const captionStat = captionPath ? fileStat(captionPath) : null;
      return {
        itemPath,
        captionPath: captionStat?.isFile() ? captionPath : null,
        itemSize: itemStat?.size || 0,
        itemMtimeMs: itemStat?.mtimeMs || 0,
        captionSize: captionStat?.isFile() ? captionStat.size : 0,
        captionMtimeMs: captionStat?.isFile() ? captionStat.mtimeMs : 0,
      };
    });
}

function fingerprintCandidates(candidates: CaptionAuditCandidate[]) {
  const hash = crypto.createHash('sha256');
  hash.update(AUDIT_VERSION);
  for (const candidate of candidates) {
    hash.update('\0');
    hash.update(candidate.itemPath);
    hash.update('\0');
    hash.update(candidate.captionPath || '');
    hash.update('\0');
    hash.update(String(candidate.itemSize));
    hash.update(':');
    hash.update(String(Math.round(candidate.itemMtimeMs)));
    hash.update(':');
    hash.update(String(candidate.captionSize));
    hash.update(':');
    hash.update(String(Math.round(candidate.captionMtimeMs)));
  }
  return hash.digest('hex');
}

function rememberAuditResult(cacheKey: string, result: CachedAudit) {
  if (!auditCache.has(cacheKey) && auditCache.size >= MAX_CACHE_ENTRIES) {
    const oldestKey = auditCache.keys().next().value;
    if (oldestKey) auditCache.delete(oldestKey);
  }
  auditCache.set(cacheKey, result);
}

async function scanRefusals(candidates: CaptionAuditCandidate[]) {
  const refusals: Record<string, string> = {};
  const candidatesWithCaptions = candidates.filter(candidate => candidate.captionPath);

  for (let index = 0; index < candidatesWithCaptions.length; index += READ_CONCURRENCY) {
    const chunk = candidatesWithCaptions.slice(index, index + READ_CONCURRENCY);
    await Promise.all(
      chunk.map(async candidate => {
        const captionPath = candidate.captionPath;
        if (!captionPath) return;
        try {
          const caption = await fsp.readFile(captionPath, 'utf8');
          if (isRefusalCaption(caption)) {
            refusals[candidate.itemPath] = caption;
          }
        } catch {
          // A caption may disappear while the dataset is being edited; the next changed fingerprint will rescan it.
        }
      }),
    );
    if (index + READ_CONCURRENCY < candidatesWithCaptions.length) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  return refusals;
}

export async function auditDatasetRefusalCaptions(datasetFolder: string): Promise<RefusalCaptionAuditResult> {
  const resolvedFolder = path.resolve(datasetFolder);
  const candidates = collectCaptionAuditCandidates(resolvedFolder);
  const datasetFingerprint = fingerprintCandidates(candidates);
  const cached = auditCache.get(resolvedFolder);

  if (cached?.datasetFingerprint === datasetFingerprint) {
    return { ...cached, cached: true };
  }

  const refusals = await scanRefusals(candidates);
  const result: CachedAudit = {
    datasetFingerprint,
    scanned: candidates.length,
    refusalCount: Object.keys(refusals).length,
    refusals,
  };
  rememberAuditResult(resolvedFolder, result);
  return { ...result, cached: false };
}
