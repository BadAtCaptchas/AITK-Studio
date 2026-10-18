'use client';

import { Images } from 'lucide-react';
import useSampleImages from '@/hooks/useSampleImages';
import { isImage } from '@/utils/basic';
import { isActiveJobStatus } from '@/utils/jobStatus';
import { getDisplayPath, getSampleThumbnailUrl } from '@/utils/media';
import type { Job } from '@/types';

export default function JobSamplePreview({ job }: { job: Pick<Job, 'id' | 'name' | 'status'> }) {
  const { sampleImages, status } = useSampleImages(job.id, isActiveJobStatus(job.status) ? 5000 : null);
  const previews = sampleImages
    .filter(sample => isImage(getDisplayPath(sample)))
    .sort((a, b) => getDisplayPath(a).localeCompare(getDisplayPath(b), undefined, { numeric: true }))
    .slice(-3)
    .reverse();

  return (
    <div className="journey-preview" aria-label="Latest run samples">
      {previews.length ? (
        previews.map(src => <img key={src} src={getSampleThumbnailUrl(src)} alt={`Sample from ${job.name}`} />)
      ) : (
        <div className="journey-preview-empty">
          <Images size={48} strokeWidth={1} aria-hidden="true" />
          <span>
            {status === 'error'
              ? 'Samples unavailable'
              : status === 'idle' || status === 'loading'
                ? 'Loading samples…'
                : 'No sample images yet'}
          </span>
        </div>
      )}
    </div>
  );
}
