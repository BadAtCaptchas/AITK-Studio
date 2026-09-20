import fs from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import type { Job } from '../types';
import { getSafeJobFolder } from './jobFolder';
import { normalizeStoragePathSetting } from './pathContainment';
import { CommandInputError } from './commandInput';

async function location(job: Job) {
  const folder = await getSafeJobFolder(job);
  const filename = await normalizeStoragePathSetting(path.join(folder, 'notes.md'), folder);
  if (!filename) throw new CommandInputError('Invalid notes path', 403);
  return { folder, filename };
}

export async function readJobNotes(job: Job): Promise<string> {
  const { filename } = await location(job);
  const stat = await fs.stat(filename).catch((error: unknown) => {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return null;
    throw error;
  });
  if (!stat) return '';
  if (!stat.isFile()) throw new CommandInputError('Invalid notes file', 400);
  if (stat.size > 1024 * 1024) throw new CommandInputError('Notes file is too large', 413);
  return fs.readFile(filename, 'utf8');
}

export async function writeJobNotes(job: Job, notes: string): Promise<void> {
  if (Buffer.byteLength(notes, 'utf8') > 1024 * 1024) throw new CommandInputError('Notes file is too large', 413);
  const { folder, filename } = await location(job);
  await fs.mkdir(folder, { recursive: true });
  const temporary = path.join(folder, `.notes-${randomUUID()}.tmp`);
  try {
    await fs.writeFile(temporary, notes, { encoding: 'utf8', flag: 'wx' });
    await fs.rename(temporary, filename);
  } finally {
    await fs.unlink(temporary).catch(() => undefined);
  }
}
