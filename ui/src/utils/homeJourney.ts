import type { DatasetSummary, Job } from '../types';

export type HomeAction = {
  stage: 'prepare' | 'train' | 'review';
  title: string;
  description: string;
  label: string;
  href: string;
  dataset?: DatasetSummary;
  job?: Job;
};

export function nextHomeAction(jobs: Job[], datasets: DatasetSummary[]): HomeAction {
  const active = jobs.find(job => ['running', 'starting', 'stopping', 'queued'].includes(job.status));
  if (active)
    return {
      stage: 'train',
      title: active.name,
      description: `Training is ${active.status}. Follow progress, samples, and checkpoints.`,
      label: 'View training',
      href: `/jobs/${encodeURIComponent(active.id)}`,
      job: active,
    };
  const attention = jobs.find(job => job.status === 'error' || (job.status === 'stopped' && job.step > 0));
  if (attention)
    return {
      stage: 'review',
      title: attention.name,
      description: attention.info || 'Review this run and choose whether to resume or start again.',
      label: 'Review run',
      href: `/jobs/${encodeURIComponent(attention.id)}`,
      job: attention,
    };
  const dataset = datasets.find(item => (item.missingCaptionCount ?? 0) > 0) || datasets[0];
  if (!dataset)
    return {
      stage: 'prepare',
      title: 'Start with your data',
      description: 'Bring in images, video, or audio, then review the media and captions before training.',
      label: 'Import data',
      href: '/datasets?action=import',
    };
  const datasetHref = `/datasets/${encodeURIComponent(dataset.name)}${dataset.worker_id && dataset.worker_id !== 'local' ? `?worker_id=${encodeURIComponent(dataset.worker_id)}` : ''}`;
  const missing = dataset.missingCaptionCount;
  const needsReview = dataset.encrypted || missing == null || missing > 0 || dataset.itemCount === 0;
  return {
    stage: needsReview ? 'prepare' : 'train',
    title: dataset.name,
    description: dataset.encrypted
      ? 'Unlock this dataset to review its media and captions.'
      : missing == null
        ? 'Open this dataset to check its contents. Caption counts are unavailable.'
        : missing > 0
          ? 'Review the remaining captions. Caption coverage describes completeness, not quality.'
          : dataset.itemCount === 0
            ? 'Add media to this dataset before training.'
            : 'Review your data and choose training settings for your model.',
    label: needsReview
      ? missing && missing > 0
        ? `Review ${missing} missing captions`
        : 'Review dataset'
      : 'Set up training',
    href: needsReview
      ? datasetHref
      : `/jobs/new?dataset=${encodeURIComponent(dataset.ref || dataset.path || dataset.name)}`,
    dataset,
  };
}
