import { getDisplayPath } from './media';

export type SampleMetadata = {
  version: 1; prompt: string; negative_prompt: string; seed: number; width: number; height: number;
  guidance_scale: number; guidance_rescale: number; sample_steps: number; network_multiplier: number;
  num_frames: number; fps: number; step: number | null; prompt_index: number; sampler: string | null; iteration_id: string | null;
  prompt_2?: string | null; negative_prompt_2?: string | null; prompt_identity?: string;
};
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const integer = (value: unknown): value is number => finite(value) && Number.isSafeInteger(value) && value >= 0;
export function parseSampleMetadata(value: unknown): SampleMetadata | null {
  if (!record(value) || value.version !== 1 || typeof value.prompt !== 'string' || value.prompt.length > 131072 ||
      !(value.negative_prompt === null || typeof value.negative_prompt === 'string' && value.negative_prompt.length <= 131072) || !integer(value.seed) ||
      !integer(value.width) || !integer(value.height) || !finite(value.guidance_scale) || !finite(value.guidance_rescale) ||
      !integer(value.sample_steps) || !finite(value.network_multiplier) || !integer(value.num_frames) || !finite(value.fps) ||
      !(value.step === null || integer(value.step)) || !integer(value.prompt_index) ||
      !(value.sampler === null || typeof value.sampler === 'string' && value.sampler.length <= 128) ||
      !(value.iteration_id === null || typeof value.iteration_id === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(value.iteration_id))) return null;
  for (const key of ['prompt_2', 'negative_prompt_2']) if (!(value[key] == null || typeof value[key] === 'string' && value[key].length <= 131072)) return null;
  if (!(value.prompt_identity === undefined || typeof value.prompt_identity === 'string' && /^[a-f0-9]{64}$/.test(value.prompt_identity))) return null;
  return { version: 1, prompt: value.prompt, negative_prompt: typeof value.negative_prompt === 'string' ? value.negative_prompt : '', seed: value.seed,
    width: value.width, height: value.height, guidance_scale: value.guidance_scale, guidance_rescale: value.guidance_rescale,
    sample_steps: value.sample_steps, network_multiplier: value.network_multiplier, num_frames: value.num_frames, fps: value.fps,
    step: typeof value.step === 'number' ? value.step : null, prompt_index: value.prompt_index,
    sampler: typeof value.sampler === 'string' ? value.sampler : null, iteration_id: typeof value.iteration_id === 'string' ? value.iteration_id : null,
    ...(value.prompt_2 !== undefined ? { prompt_2: typeof value.prompt_2 === 'string' ? value.prompt_2 : null } : {}),
    ...(value.negative_prompt_2 !== undefined ? { negative_prompt_2: typeof value.negative_prompt_2 === 'string' ? value.negative_prompt_2 : null } : {}),
    ...(typeof value.prompt_identity === 'string' ? { prompt_identity: value.prompt_identity } : {}) };
}

export function parseSampleMetadataMap(value: unknown, samples: string[]): Record<string, SampleMetadata> {
  const result: Record<string, SampleMetadata> = {};
  if (!record(value)) return result;
  for (const sample of samples) {
    const metadata = parseSampleMetadata(value[sample]);
    if (metadata) result[sample] = metadata;
  }
  return result;
}

export function sampleIdentity(sample: string, metadata?: SampleMetadata) {
  const filename = getDisplayPath(sample).split(/[\\/]/).pop() || '';
  const match = /^(\d+)__(\d+)_(\d+)\.[^.]+$/.exec(filename);
  const step = metadata ? metadata.step : match ? Number(match[2]) : null;
  const promptIndex = metadata ? metadata.prompt_index : match ? Number(match[3]) : null;
  const group = metadata?.iteration_id ? `iteration:${metadata.iteration_id}` : step !== null ? `step:${step}` : `file:${sample}`;
  return { filename, step, promptIndex, group };
}

export function groupSampleRows(samples: string[], metadata: Record<string, SampleMetadata>): string[][] {
  const groups = new Map<string, string[]>();
  for (const sample of samples) {
    const { group } = sampleIdentity(sample, metadata[sample]);
    const row = groups.get(group) || [];
    row.push(sample);
    groups.set(group, row);
  }
  return [...groups.values()].map(row => row.sort((a, b) =>
    (sampleIdentity(a, metadata[a]).promptIndex ?? 0) - (sampleIdentity(b, metadata[b]).promptIndex ?? 0)));
}

export function adjacentSample(samples: string[], metadata: Record<string, SampleMetadata>, current: string, direction: 'up' | 'down' | 'left' | 'right'): string | undefined {
  const rows = groupSampleRows(samples, metadata);
  const rowIndex = rows.findIndex(row => row.includes(current));
  if (rowIndex < 0) return undefined;
  if (direction === 'left' || direction === 'right') return rows[rowIndex][rows[rowIndex].indexOf(current) + (direction === 'left' ? -1 : 1)];
  const promptIndex = sampleIdentity(current, metadata[current]).promptIndex;
  if (promptIndex === null) return undefined;
  const delta = direction === 'up' ? -1 : 1;
  const currentMetadata = metadata[current];
  const promptIdentity = (item: SampleMetadata) => item.prompt_identity || JSON.stringify([item.prompt, item.negative_prompt, item.prompt_2 ?? null, item.negative_prompt_2 ?? null]);
  for (let index = rowIndex + delta; index >= 0 && index < rows.length; index += delta) {
    const match = rows[index].find(sample => {
      const candidate = metadata[sample];
      if (currentMetadata) return !!candidate && promptIdentity(candidate) === promptIdentity(currentMetadata);
      return !candidate && sampleIdentity(sample).promptIndex === promptIndex;
    });
    if (match) return match;
  }
  return undefined;
}
