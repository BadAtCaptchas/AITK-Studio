import { randomUUID } from 'crypto';
import { db } from './db';
import {
  buildIdeogramComfyWorkflow,
  cloneIdeogramWorkflowState,
  type IdeogramWorkflowState,
} from '../utils/ideogramWorkflow';
import type { ComfyImageRef } from './externalComfy';

export const IDEOGRAM_WORKFLOW_HISTORY_KEY = 'IDEOGRAM_WORKFLOW_HISTORY';
const RECENT_HISTORY_LIMIT = 80;

export type IdeogramWorkflowHistoryStatus = 'queued' | 'completed' | 'error' | 'imported';

export type IdeogramWorkflowHistoryEntry = {
  id: string;
  title: string;
  promptId: string;
  serverUrl: string;
  state: IdeogramWorkflowState;
  workflow: Record<string, unknown>;
  images: ComfyImageRef[];
  favorite: boolean;
  status: IdeogramWorkflowHistoryStatus;
  createdAt: string;
  updatedAt: string;
  aspectRatio: string;
  seed: number;
  steps: number;
  cfg: number;
};

export type IdeogramWorkflowHistoryInput = {
  id?: string;
  title?: string;
  promptId?: string;
  serverUrl?: string;
  state: IdeogramWorkflowState;
  workflow?: Record<string, unknown>;
  images?: ComfyImageRef[];
  favorite?: boolean;
  status?: IdeogramWorkflowHistoryStatus;
};

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function cleanText(value: unknown, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function cleanDate(value: unknown, fallback = new Date().toISOString()) {
  const date = typeof value === 'string' || typeof value === 'number' ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime()) ? date.toISOString() : fallback;
}

function cleanStatus(value: unknown): IdeogramWorkflowHistoryStatus {
  return value === 'queued' || value === 'completed' || value === 'error' || value === 'imported' ? value : 'completed';
}

function cleanImages(value: unknown): ComfyImageRef[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap(image => {
    if (!isRecord(image) || typeof image.filename !== 'string' || !image.filename.trim()) return [];
    return [
      {
        filename: image.filename,
        subfolder: typeof image.subfolder === 'string' ? image.subfolder : '',
        type: typeof image.type === 'string' ? image.type : 'output',
      },
    ];
  });
}

function historyTitle(state: IdeogramWorkflowState, fallback = 'Ideogram workflow') {
  const text = state.highLevelDescription.trim().replace(/\s+/g, ' ');
  if (!text) return fallback;
  return text.length > 76 ? `${text.slice(0, 75)}...` : text;
}

function normalizeHistoryEntry(raw: unknown): IdeogramWorkflowHistoryEntry | null {
  if (!isRecord(raw) || !isRecord(raw.state)) return null;
  const state = cloneIdeogramWorkflowState(raw.state as IdeogramWorkflowState);
  const now = new Date().toISOString();
  const workflow = isRecord(raw.workflow) ? raw.workflow : buildIdeogramComfyWorkflow(state);
  return {
    id: cleanText(raw.id) || randomUUID(),
    title: cleanText(raw.title) || historyTitle(state),
    promptId: cleanText(raw.promptId || raw.prompt_id),
    serverUrl: cleanText(raw.serverUrl || raw.server_url),
    state,
    workflow,
    images: cleanImages(raw.images),
    favorite: Boolean(raw.favorite),
    status: cleanStatus(raw.status),
    createdAt: cleanDate(raw.createdAt || raw.created_at, now),
    updatedAt: cleanDate(raw.updatedAt || raw.updated_at, now),
    aspectRatio: state.aspectRatio,
    seed: state.seed,
    steps: state.steps,
    cfg: state.guiderCfg,
  };
}

async function readHistoryStore() {
  const row = await db.settings.get(IDEOGRAM_WORKFLOW_HISTORY_KEY);
  if (!row?.value) return [];
  try {
    const parsed = JSON.parse(row.value);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizeHistoryEntry).filter((entry): entry is IdeogramWorkflowHistoryEntry => entry !== null);
  } catch {
    return [];
  }
}

async function writeHistoryStore(entries: IdeogramWorkflowHistoryEntry[]) {
  await db.settings.upsert(IDEOGRAM_WORKFLOW_HISTORY_KEY, JSON.stringify(entries));
}

function pruneHistory(entries: IdeogramWorkflowHistoryEntry[]) {
  const sorted = [...entries].sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  const favorites = sorted.filter(entry => entry.favorite);
  const recents = sorted.filter(entry => !entry.favorite).slice(0, RECENT_HISTORY_LIMIT);
  return [...favorites, ...recents].sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
}

export async function listIdeogramWorkflowHistory() {
  return pruneHistory(await readHistoryStore());
}

export async function upsertIdeogramWorkflowHistoryEntry(input: IdeogramWorkflowHistoryInput) {
  const state = cloneIdeogramWorkflowState(input.state);
  const entries = await readHistoryStore();
  const existing = entries.find(entry => (input.id && entry.id === input.id) || (input.promptId && entry.promptId === input.promptId));
  const now = new Date().toISOString();
  const entry: IdeogramWorkflowHistoryEntry = {
    id: existing?.id || input.id || randomUUID(),
    title: input.title?.trim() || existing?.title || historyTitle(state),
    promptId: input.promptId?.trim() || existing?.promptId || '',
    serverUrl: input.serverUrl?.trim() || existing?.serverUrl || '',
    state,
    workflow: input.workflow || buildIdeogramComfyWorkflow(state),
    images: input.images || existing?.images || [],
    favorite: input.favorite ?? existing?.favorite ?? false,
    status: input.status || existing?.status || 'completed',
    createdAt: existing?.createdAt || now,
    updatedAt: now,
    aspectRatio: state.aspectRatio,
    seed: state.seed,
    steps: state.steps,
    cfg: state.guiderCfg,
  };
  const nextEntries = pruneHistory([entry, ...entries.filter(item => item.id !== entry.id)]);
  await writeHistoryStore(nextEntries);
  return { entry, entries: nextEntries };
}

export async function setIdeogramWorkflowHistoryFavorite(id: string, favorite: boolean) {
  const entries = await readHistoryStore();
  const nextEntries = pruneHistory(
    entries.map(entry => (entry.id === id ? { ...entry, favorite, updatedAt: new Date().toISOString() } : entry)),
  );
  await writeHistoryStore(nextEntries);
  const entry = nextEntries.find(item => item.id === id) || null;
  return { entry, entries: nextEntries };
}

export async function deleteIdeogramWorkflowHistoryEntry(id: string) {
  const entries = await readHistoryStore();
  const nextEntries = entries.filter(entry => entry.id !== id);
  await writeHistoryStore(nextEntries);
  return { deleted: nextEntries.length !== entries.length, entries: nextEntries };
}
