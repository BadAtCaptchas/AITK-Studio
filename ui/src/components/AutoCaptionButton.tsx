import { useEffect, useState } from 'react';
import { Button } from '@headlessui/react';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useJobByRef from '@/hooks/useJobByRef';
import Link from 'next/link';
import { Loader2, Sparkles } from 'lucide-react';
import classNames from 'classnames';
const ACTIVE_CAPTION_STATUSES = new Set(['running', 'queued', 'stopping']);
const EDIT_LOCK_CAPTION_STATUSES = new Set(['running']);
type AutoCaptionButtonProps = {
  datasetPath: string;
  datasetName: string;
  setIsAutoCaptioning?: (isAutoCaptioning: boolean) => void;
  encryptedDatasetKeyB64?: string;
  rootCaption?: string | null;
  className?: string;
  idleLabel?: string;
};
export default function AutoCaptionButton({
  datasetPath,
  datasetName,
  setIsAutoCaptioning,
  encryptedDatasetKeyB64,
  rootCaption,
  className,
  idleLabel = 'Auto Caption',
}: AutoCaptionButtonProps) {
  const [reloadInterval, setReloadInterval] = useState<number | null>(null);
  const { job, refreshJob } = useJobByRef(datasetPath, reloadInterval, 'caption');
  const isActiveCaptionJob = !!job && job.job_type === 'caption' && ACTIVE_CAPTION_STATUSES.has(job.status);
  const isCaptionEditLocked = !!job && job.job_type === 'caption' && EDIT_LOCK_CAPTION_STATUSES.has(job.status);
  useEffect(() => {
    setReloadInterval(isActiveCaptionJob ? 5000 : null);
  }, [isActiveCaptionJob]);
  useEffect(() => {
    if (setIsAutoCaptioning) {
      setIsAutoCaptioning(isCaptionEditLocked);
    }
  }, [isCaptionEditLocked, setIsAutoCaptioning]);
  if (isActiveCaptionJob && job) {
    const label = job.status === 'queued' ? 'Auto Caption Queued...' : 'Auto Captioning...';
    return (
      <Link
        href={`/jobs/${job.id}`}
        className={classNames(
          'inline-flex items-center gap-1.5 rounded-md bg-gray-500 px-3 py-1 text-white',
          className,
        )}
      >
        {job.status === 'running' && <Loader2 className="w-4 h-4 animate-spin" />}
        {label}
      </Link>
    );
  }
  return (
    <Button
      className={classNames(
        'inline-flex items-center gap-2 rounded-md bg-brand-500 px-3 py-1 text-[var(--brand-ink)]',
        className,
      )}
      onClick={() =>
        openCaptionDatasetModal(
          datasetPath,
          () => {
            refreshJob();
          },
          { encryptedDatasetKeyB64, datasetName, rootCaption },
        )
      }
    >
      <Sparkles className="h-4 w-4" />
      {idleLabel}
    </Button>
  );
}
