import { useEffect, useState } from 'react';
import { Button } from '@headlessui/react';
import { CaptionDatasetModal, openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useJobByRef from '@/hooks/useJobByRef';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';

const ACTIVE_CAPTION_STATUSES = new Set(['running', 'queued', 'stopping']);
const EDIT_LOCK_CAPTION_STATUSES = new Set(['running']);

type AutoCaptionButtonProps = {
  datasetPath: string;
  datasetName: string;
  projectID?: string | null;
  setIsAutoCaptioning?: (isAutoCaptioning: boolean) => void;
  encryptedDatasetKeyB64?: string;
  rootCaption?: string | null;
};

export default function AutoCaptionButton({
  datasetPath,
  datasetName,
  projectID = null,
  setIsAutoCaptioning,
  encryptedDatasetKeyB64,
  rootCaption,
}: AutoCaptionButtonProps) {
  const [reloadInterval, setReloadInterval] = useState<number | null>(null);
  const { job, refreshJob } = useJobByRef(datasetPath, reloadInterval, 'caption', projectID);
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
      <Link href={`/jobs/${job.id}`} className="text-white bg-gray-400 px-3 py-1 rounded-md mr-2 inline-flex items-center gap-1.5">
        {job.status === 'running' && <Loader2 className="w-4 h-4 animate-spin" />}
        {label}
      </Link>
    );
  }
  return (
    <Button
      className="text-white bg-blue-600 px-3 py-1 rounded-md mr-2"
      onClick={() =>
        openCaptionDatasetModal(
          datasetPath,
          () => {
            refreshJob();
          },
          { encryptedDatasetKeyB64, projectID, datasetName, rootCaption },
        )
      }
    >
      Auto Caption
    </Button>
  );
}
