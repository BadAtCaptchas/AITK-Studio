'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@headlessui/react';
import { Upload } from 'lucide-react';
import { SelectInput } from '@/components/formInputs';
import useGPUInfo from '@/hooks/useGPUInfo';
import { importTrainingJob } from '@/utils/jobs';

export default function Dashboard() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);

  useEffect(() => {
    if (isGPUInfoLoaded && gpuIDs === null && gpuList.length > 0) {
      setGpuIDs(`${gpuList[0].index}`);
    }
  }, [gpuIDs, gpuList, isGPUInfoLoaded]);

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || isImporting) return;

    setIsImporting(true);
    try {
      const result = await importTrainingJob(file, gpuIDs);
      if (result.warnings?.length) {
        alert(`Import completed with warnings:\n\n${result.warnings.join('\n')}`);
      }
      router.push(`/jobs/${result.job.id}`);
    } catch (error) {
      console.error('Error importing training job:', error);
      alert('Failed to import training job. Please check the archive and try again.');
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        {gpuList.length > 0 && (
          <div className="mr-2">
            <SelectInput
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
        <div className="mr-2">
          <Button
            className="text-white bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-md inline-flex items-center gap-2 disabled:opacity-60"
            onClick={() => fileInputRef.current?.click()}
            disabled={isImporting}
          >
            <Upload className="w-4 h-4" />
            {isImporting ? 'Importing...' : 'Import Training Job'}
          </Button>
        </div>
        <div>
          <Link href="/jobs/new" className="text-white bg-slate-600 px-3 py-1 rounded-md">
            New Training Job
          </Link>
        </div>
      </TopBar>
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,.aitk"
        className="hidden"
        onChange={handleFileSelected}
      />
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
