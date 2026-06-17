import { useEffect, useState } from 'react';
import { Button } from '@headlessui/react';
import { CaptionDatasetModal, openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import useJobByRef from '@/hooks/useJobByRef';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';

type AutoCaptionButtonProps = {
  datasetPath: string;
  projectID?: string | null;
  setIsAutoCaptioning?: (isAutoCaptioning: boolean) => void;
  encryptedDatasetKeyB64?: string;
};

export default function AutoCaptionButton({ datasetPath, projectID = null, setIsAutoCaptioning, encryptedDatasetKeyB64 }: AutoCaptionButtonProps) {
  const [reloadInterval, setReloadInterval] = useState<number | null>(null);
  const { job, refreshJob } = useJobByRef(datasetPath, reloadInterval);
  const isCaptioning = !!job && (job.status === 'running' || job.status === 'queued' || job.status === 'stopping');

  useEffect(() => {
    setReloadInterval(isCaptioning ? 5000 : null);
  }, [isCaptioning]);

  useEffect(() => {
    if (setIsAutoCaptioning) {
      setIsAutoCaptioning(isCaptioning);
    }
  }, [isCaptioning, setIsAutoCaptioning]);
  
  if (isCaptioning && job) {
    return (
      <Link href={`/jobs/${job.id}`} className="text-white bg-gray-400 px-3 py-1 rounded-md mr-2 inline-flex items-center gap-1.5">
        <Loader2 className="w-4 h-4 animate-spin" />
        Auto Captioning...
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
          { encryptedDatasetKeyB64, projectID },
        )
      }
    >
      Auto Caption
    </Button>
  );
}
