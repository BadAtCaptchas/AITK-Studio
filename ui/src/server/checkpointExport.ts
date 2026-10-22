import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { createHash } from 'crypto';
import { isRecord } from './commandInput';
import { isPathWithinRoot } from './pathContainment';

export const CHECKPOINT_GENERATIONS = '.aitk-checkpoints';
const CHECKPOINT_MANIFEST = 'checkpoint.json';

export type CheckpointExportFile = { sourcePath: string; relativePath: string; size: number };
type GenerationMember = { path: string; size: number; sha256: string };
type Generation = {
  directory: string;
  relativeDirectory: string;
  step: number;
  created: number;
  primary: string;
  members: GenerationMember[];
  optimizer: string | null;
  manifestSize: number;
};
export type CheckpointExportSelection = {
  relativePath: string | null;
  step: number | null;
  optimizerIncluded: boolean;
  files: CheckpointExportFile[];
  excludedTopLevelNames: Set<string>;
  warnings: string[];
};

function memberPath(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && !/[\\:\x00-\x1f]/.test(value)
    && value.split('/').every(part => !!part && part !== '.' && part !== '..');
}

async function plainPath(root: string, relative: string) {
  if (!memberPath(relative)) throw new Error('Invalid checkpoint member path');
  const destination = path.resolve(root, ...relative.split('/'));
  if (!isPathWithinRoot(root, destination)) throw new Error('Checkpoint member escapes its generation');
  let current = root;
  for (const part of relative.split('/')) {
    current = path.join(current, part);
    if ((await fsp.lstat(current)).isSymbolicLink()) throw new Error('Checkpoint members cannot be links');
  }
  if (!isPathWithinRoot(await fsp.realpath(root), await fsp.realpath(destination)))
    throw new Error('Checkpoint member escapes its generation');
  return destination;
}

async function readGeneration(directory: string): Promise<Generation> {
  if (!(await fsp.lstat(directory)).isDirectory() || (await fsp.lstat(directory)).isSymbolicLink())
    throw new Error('Checkpoint generation must be a plain directory');
  const marker = await plainPath(directory, CHECKPOINT_MANIFEST);
  const markerStat = await fsp.stat(marker);
  if (!markerStat.isFile() || markerStat.size > 16 * 1024 * 1024) throw new Error('Invalid checkpoint manifest size');
  const value: unknown = JSON.parse(await fsp.readFile(marker, 'utf8'));
  if (!isRecord(value) || value.version !== 1) throw new Error('Unsupported checkpoint manifest version');
  if (typeof value.step !== 'number' || !Number.isSafeInteger(value.step) || value.step < 0
    || typeof value.created_ns !== 'number' || !Number.isInteger(value.created_ns) || value.created_ns < 0
    || !memberPath(value.primary) || !Array.isArray(value.files) || value.files.length === 0 || value.files.length > 100_000)
    throw new Error('Invalid checkpoint manifest');
  const members: GenerationMember[] = [];
  const names = new Set<string>();
  for (const entry of value.files) {
    if (!isRecord(entry) || !memberPath(entry.path) || entry.path === CHECKPOINT_MANIFEST || names.has(entry.path)
      || typeof entry.size !== 'number' || !Number.isSafeInteger(entry.size) || entry.size < 0
      || typeof entry.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(entry.sha256))
      throw new Error('Invalid checkpoint member');
    const target = await plainPath(directory, entry.path);
    const stat = await fsp.stat(target);
    if (!stat.isFile() || stat.size !== entry.size) throw new Error('Incomplete checkpoint member');
    members.push({ path: entry.path, size: entry.size, sha256: entry.sha256 });
    names.add(entry.path);
  }
  const models = value.model_files;
  if (!Array.isArray(models) || !models.length || models.some(item => typeof item !== 'string' || !names.has(item) || item === 'optimizer.pt')
    || new Set(models).size !== models.length || models.length !== members.filter(item => item.path !== 'optimizer.pt').length)
    throw new Error('Invalid checkpoint model artifacts');
  if (value.optimizer !== null && (value.optimizer !== 'optimizer.pt' || !names.has('optimizer.pt')))
    throw new Error('Invalid checkpoint optimizer');
  if (names.has('optimizer.pt') !== (value.optimizer === 'optimizer.pt')) throw new Error('Unpaired checkpoint optimizer');
  const primary = await fsp.stat(await plainPath(directory, value.primary));
  if (primary.isFile() ? !models.includes(value.primary) : !primary.isDirectory() || !models.some(item => item.startsWith(`${value.primary}/`)))
    throw new Error('Invalid primary checkpoint');
  const actual = new Set<string>();
  async function walk(current: string) {
    for (const entry of await fsp.readdir(current, { withFileTypes: true })) {
      const filename = path.join(current, entry.name);
      if (entry.isSymbolicLink()) throw new Error('Checkpoint members cannot be links');
      if (entry.isDirectory()) await walk(filename);
      else if (entry.isFile()) actual.add(path.relative(directory, filename).replace(/\\/g, '/'));
      else throw new Error('Checkpoint contains a special file');
    }
  }
  await walk(directory);
  actual.delete(CHECKPOINT_MANIFEST);
  if (actual.size !== names.size || [...actual].some(name => !names.has(name))) throw new Error('Checkpoint contains unlisted members');
  return {
    directory, relativeDirectory: `${CHECKPOINT_GENERATIONS}/${path.basename(directory)}`,
    step: value.step, created: value.created_ns, primary: value.primary, members,
    optimizer: value.optimizer === 'optimizer.pt' ? 'optimizer.pt' : null, manifestSize: markerStat.size,
  };
}

async function verifyGeneration(generation: Generation, checkCanceled: () => void) {
  for (const entry of generation.members) {
    checkCanceled();
    const hash = createHash('sha256');
    for await (const chunk of fs.createReadStream(await plainPath(generation.directory, entry.path))) {
      checkCanceled();
      hash.update(chunk);
    }
    if (hash.digest('hex') !== entry.sha256) throw new Error('Checkpoint checksum mismatch');
  }
}

/** Canonical generations and portable aliases must originate from the same immutable save. */
export async function selectCheckpointExport(
  trainingFolder: string, dbStep: number, mode: 'latest' | 'all', checkCanceled: () => void = () => {},
): Promise<CheckpointExportSelection> {
  const result: CheckpointExportSelection = {
    relativePath: null, step: null, optimizerIncluded: false, files: [],
    excludedTopLevelNames: new Set([CHECKPOINT_GENERATIONS, 'optimizer.pt']), warnings: [],
  };
  if (!fs.existsSync(trainingFolder)) return result;
  const entries = await fsp.readdir(trainingFolder, { withFileTypes: true });
  const generationRoot = path.join(trainingFolder, CHECKPOINT_GENERATIONS);
  const generations: Generation[] = [];
  let hadCommitted = false;
  if (entries.some(entry => entry.name === CHECKPOINT_GENERATIONS)) {
    const rootStat = await fsp.lstat(generationRoot);
    if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) throw new Error('Checkpoint generation folder cannot be a link');
    for (const entry of await fsp.readdir(generationRoot, { withFileTypes: true })) {
      checkCanceled();
      if (entry.name.startsWith('.tmp-')) continue;
      hadCommitted = true;
      try {
        const generation = await readGeneration(path.join(generationRoot, entry.name));
        generations.push(generation);
        for (const member of generation.members) result.excludedTopLevelNames.add(member.path.split('/')[0]);
      } catch {
        result.warnings.push(`Skipped incomplete or unsupported checkpoint generation: ${entry.name}`);
      }
    }
  }
  // Portable weights are conveniences, never an alternate commit when generations exist.
  for (const entry of entries) {
    if (entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')
      || entry.isDirectory() && fs.existsSync(path.join(trainingFolder, entry.name, 'aitk_meta.yaml')))
      result.excludedTopLevelNames.add(entry.name);
  }
  generations.sort((a, b) => b.step - a.step || b.created - a.created);
  const selected: Generation[] = [];
  for (const generation of generations) {
    checkCanceled();
    try {
      await verifyGeneration(generation, checkCanceled);
      selected.push(generation);
      if (mode === 'latest') break;
    } catch (error) {
      checkCanceled();
      result.warnings.push(`Skipped corrupt checkpoint generation: ${path.basename(generation.directory)}`);
      if (!(error instanceof Error)) throw error;
    }
  }
  if (hadCommitted && selected.length === 0) throw new Error('No valid committed checkpoint remains; cannot export a resumable job');
  const latest = selected[0];
  if (latest) {
    result.relativePath = `${latest.relativeDirectory}/${latest.primary}`;
    result.step = latest.step;
    result.optimizerIncluded = latest.optimizer !== null;
    for (const generation of selected) {
      result.files.push({
        sourcePath: path.join(generation.directory, CHECKPOINT_MANIFEST),
        relativePath: `${generation.relativeDirectory}/${CHECKPOINT_MANIFEST}`, size: generation.manifestSize,
      });
      for (const member of generation.members) result.files.push({
        sourcePath: path.join(generation.directory, member.path),
        relativePath: `${generation.relativeDirectory}/${member.path}`, size: member.size,
      });
    }
    // Older workers resume from these aliases. Never read aliases that a running save may replace.
    for (const member of latest.members) result.files.push({
      sourcePath: path.join(latest.directory, member.path), relativePath: member.path, size: member.size,
    });
    return result;
  }
  const legacy = entries.filter(entry => entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')
    || entry.isDirectory() && fs.existsSync(path.join(trainingFolder, entry.name, 'aitk_meta.yaml')))
    .map(entry => ({ name: entry.name, directory: entry.isDirectory(),
      step: Number(entry.name.match(/_(\d+)(?:_(?:high|low)_noise)?(?:\.safetensors)?$/)?.[1] ?? dbStep) }));
  legacy.sort((a, b) => b.step - a.step || a.name.localeCompare(b.name));
  const legacyLatest = legacy[0];
  if (legacyLatest) {
    result.relativePath = legacyLatest.name;
    result.step = legacyLatest.step;
    const splitBase = legacyLatest.name.replace(/_(?:high|low)_noise\.safetensors$/, '');
    for (const candidate of legacy) {
      if (mode === 'latest' && candidate.name !== legacyLatest.name
        && !(splitBase !== legacyLatest.name && candidate.name.replace(/_(?:high|low)_noise\.safetensors$/, '') === splitBase)) continue;
      const sourcePath = path.join(trainingFolder, candidate.name);
      if (candidate.directory) {
        async function appendDirectory(current: string) {
          for (const entry of await fsp.readdir(current, { withFileTypes: true })) {
            checkCanceled();
            const member = path.join(current, entry.name);
            if (entry.isSymbolicLink()) throw new Error('Checkpoint members cannot be links');
            if (entry.isDirectory()) await appendDirectory(member);
            else if (entry.isFile()) result.files.push({
              sourcePath: member, relativePath: path.relative(trainingFolder, member).replace(/\\/g, '/'),
              size: (await fsp.stat(member)).size,
            });
          }
        }
        await appendDirectory(sourcePath);
      } else result.files.push({ sourcePath, relativePath: candidate.name, size: (await fsp.stat(sourcePath)).size });
    }
    const optimizer = path.join(trainingFolder, 'optimizer.pt');
    if (fs.existsSync(optimizer) && (await fsp.lstat(optimizer)).isFile()) {
      result.files.push({ sourcePath: optimizer, relativePath: 'optimizer.pt', size: (await fsp.stat(optimizer)).size });
      result.optimizerIncluded = true;
    }
  }
  return result;
}
