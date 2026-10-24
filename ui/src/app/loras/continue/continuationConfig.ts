import type { JobConfig } from '@/types';
import { defaultJobConfig, migrateJobConfig } from '@/app/jobs/new/jobConfig';
import { handleModelArchChange } from '@/app/jobs/new/utils';
import { setNestedValue } from '@/utils/hooks';

export function initializeContinuationConfig(config: JobConfig): JobConfig {
  const next = structuredClone(config);
  const process = next.config.process[0];
  if (!['krea2', 'krea2:turbo'].includes(process.model.arch)) throw new Error('Continue LoRA supports Krea 2 Raw and Turbo. Import a Krea 2 configuration.');
  if (!process.network) throw new Error('Continue LoRA requires a LoRA network configuration.');
  if (!process.network.continuation) {
    process.network.type = 'lora';
    process.network.continuation = { version: 1, source_sha256: '', training_base: process.model.arch === 'krea2:turbo' ? 'turbo' : 'raw', deployment_base: 'turbo' };
    process.train.source_preservation = { mode: 'none' };
    delete process.train.start_step;
  }
  return next;
}

export function changeContinuationBase(config: JobConfig, arch: string): JobConfig {
  if (arch !== 'krea2' && arch !== 'krea2:turbo') throw new Error('Select Krea 2 Raw or Turbo.');
  let next = structuredClone(config);
  handleModelArchChange(next.config.process[0].model.arch, arch, next, (value: unknown, key: string) => {
    next = setNestedValue(next, value, key);
  });
  next = initializeContinuationConfig(next);
  const source = next.config.process[0].network?.continuation;
  if (source) source.training_base = arch === 'krea2:turbo' ? 'turbo' : 'raw';
  return next;
}

export function defaultContinuationConfig(): JobConfig {
  const config = changeContinuationBase(migrateJobConfig(structuredClone(defaultJobConfig)), 'krea2');
  config.config.name = 'continued_lora';
  return config;
}
