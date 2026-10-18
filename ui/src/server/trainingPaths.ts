import type { Job } from '../types';
import { getTrainingFolder } from './settings';
import { isRecord } from './commandInput';

function mapJobProcesses(
  jobConfig: unknown,
  transform: (process: Record<string, unknown>) => Record<string, unknown>,
): unknown {
  if (!isRecord(jobConfig) || !isRecord(jobConfig.config) || !Array.isArray(jobConfig.config.process)) return jobConfig;
  return {
    ...jobConfig,
    config: {
      ...jobConfig.config,
      process: jobConfig.config.process.map((process: unknown) => (isRecord(process) ? transform(process) : process)),
    },
  };
}

/** The executing worker owns its output location, including across config edits. */
export async function prepareLocalJobConfig(jobConfig: unknown, existing?: Pick<Job, 'job_config'>): Promise<unknown> {
  const trainingRoot = existing ? await getJobTrainingRoot(existing) : await getTrainingFolder();
  return mapJobProcesses(jobConfig, process => ({ ...process, training_folder: trainingRoot }));
}

/** Controller paths must not overwrite paths established by the worker's import. */
export function prepareRemoteJobConfigEdit(jobConfig: unknown): unknown {
  return mapJobProcesses(jobConfig, process => {
    const result = { ...process };
    delete result.training_folder;
    delete result.sqlite_db_path;
    return result;
  });
}

/** Existing runs keep the configured output root saved with their configuration. */
export async function getJobTrainingRoot(job: Pick<Job, 'job_config'>): Promise<string> {
  try {
    const parsed: unknown = JSON.parse(job.job_config);
    if (parsed && typeof parsed === 'object' && 'config' in parsed) {
      const config = parsed.config;
      if (config && typeof config === 'object' && 'process' in config && Array.isArray(config.process)) {
        const first: unknown = config.process[0];
        if (
          first &&
          typeof first === 'object' &&
          'training_folder' in first &&
          typeof first.training_folder === 'string' &&
          first.training_folder.trim()
        )
          return first.training_folder;
      }
    }
  } catch {
    /* Older or incomplete configurations use the configured global root. */
  }
  return getTrainingFolder();
}
