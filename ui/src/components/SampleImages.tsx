import { deleteSample } from '@/utils/sampleActions';
import { useMemo, useState, useRef, useCallback, useEffect } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import type { Job, JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan, LuTrash2 } from 'react-icons/lu';
import { Button, Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import { getDownloadUrl } from '@/utils/media';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';
import { openConfirm } from './ConfirmModal';
import { groupSampleRows } from '@/utils/sampleMetadata';

interface SampleImagesMenuProps {
  job?: Job | null;
}

function getSampleConfigFromJob(job: Job) {
  const jobConfig = JSON.parse(job.job_config) as JobConfig;
  const processConfig: any = jobConfig.config.process[0];
  if (processConfig.sample) {
    return processConfig.sample;
  }
  if (processConfig.generate) {
    const prompts = Array.isArray(processConfig.generate.prompts) ? processConfig.generate.prompts : [];
    return {
      sampler: processConfig.generate.sampler || 'flowmatch',
      sample_every: 1,
      width: processConfig.generate.width || 512,
      height: processConfig.generate.height || 512,
      samples: prompts.map((prompt: string) => ({
        prompt,
        width: processConfig.generate.width,
        height: processConfig.generate.height,
        neg: processConfig.generate.neg || '',
        seed: processConfig.generate.seed,
        guidance_scale: processConfig.generate.guidance_scale,
        sample_steps: processConfig.generate.sample_steps,
      })),
      neg: processConfig.generate.neg || '',
      seed: processConfig.generate.seed ?? -1,
      walk_seed: false,
      guidance_scale: processConfig.generate.guidance_scale || 4,
      sample_steps: processConfig.generate.sample_steps || 20,
      num_frames: 1,
      fps: 16,
    };
  }
  return null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);

  if (job?.worker_id && job.worker_id !== 'local') {
    return null;
  }

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobID: job?.id,
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = res.data.downloadUrl || getDownloadUrl(zipPath);
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <Button
      onClick={downloadZip}
      className={classNames(`px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700`, {
        'opacity-50 cursor-not-allowed': isZipping,
      })}
    >
      {isZipping ? (
        <LuLoader className="animate-spin inline-block mr-2" />
      ) : (
        <FaDownload className="inline-block mr-2" />
      )}
      {isZipping ? 'Preparing' : 'Download'}
    </Button>
  );
};

interface SampleImagesProps {
  job: Job;
  layout?: 'page' | 'panel';
}

export default function SampleImages({ job, layout = 'page' }: SampleImagesProps) {
  return <JobSampleImages key={job.id} job={job} layout={layout} />;
}

function JobSampleImages({ job, layout = 'page' }: SampleImagesProps) {
  const { sampleImages, sampleMetadata, status, refreshSampleImages } = useSampleImages(job.id, 5000, true);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState(false);
  // multi-select for bulk delete: shift-click ranges from the anchor, ctrl/cmd-click toggles
  const [selectedSet, setSelectedSet] = useState<Set<string>>(() => new Set());
  const anchorIdxRef = useRef<number | null>(null);
  // selection as it was when the anchor was set; shift-click ranges are rebuilt on top of it, not accumulated
  const baseSetRef = useRef<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [preservationOpen, setPreservationOpen] = useState(false);
  const [preservationName, setPreservationName] = useState('');
  const [preservationBusy, setPreservationBusy] = useState(false);
  const [preservationMessage, setPreservationMessage] = useState('');
  const preservationRequest = useRef('');
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const sampleConfig = getSampleConfigFromJob(job);
      if (!sampleConfig) return 1;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples?.length || 0;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  const rows = useMemo(() => groupSampleRows(sampleImages, sampleMetadata), [sampleImages, sampleMetadata]);

  const selectedPaths = useMemo(() => sampleImages.filter(sample => selectedSet.has(sample)), [sampleImages, selectedSet]);
  const isSelecting = selectionMode || selectedPaths.length > 0;
  const createPreservation = async () => {
    setPreservationBusy(true);
    setPreservationMessage('');
    try {
      if (!preservationRequest.current) preservationRequest.current = crypto.randomUUID();
      const samples = selectedPaths.map(sample => decodeURIComponent(sample.split('/').pop() || ''));
      const { data } = await apiClient.post<unknown>('/api/training/lora/preservation', {
        job_id: job.id, name: preservationName, request_id: preservationRequest.current, samples,
      });
      if (!data || typeof data !== 'object' || !('name' in data) || typeof data.name !== 'string') throw new Error('Invalid preservation response.');
      setPreservationMessage(`Created ${data.name}. Select this dataset in the continuation job’s preservation settings.`);
      setPreservationOpen(false);
    } catch (error: unknown) {
      const response = error && typeof error === 'object' && 'response' in error ? error.response : null;
      const data = response && typeof response === 'object' && 'data' in response ? response.data : null;
      setPreservationMessage(data && typeof data === 'object' && 'error' in data && typeof data.error === 'string' ? data.error : error instanceof Error ? error.message : 'Could not create preservation dataset.');
    } finally { setPreservationBusy(false); }
  };
  const clearSelection = useCallback(() => {
    setSelectedSet(new Set());
    anchorIdxRef.current = null;
    baseSetRef.current = new Set();
  }, []);
  const setSampleSelected = useCallback((sample: string, checked: boolean) => {
    setSelectedSet(previous => {
      const next = new Set(previous);
      if (checked) next.add(sample);
      else next.delete(sample);
      return next;
    });
    anchorIdxRef.current = null;
    baseSetRef.current = new Set();
  }, []);

  const handleCardClick = useCallback(
    (sample: string, e: React.MouseEvent) => {
      if (deleteProgress) return;
      const isRange = e.shiftKey;
      const isToggle = e.ctrlKey || e.metaKey || isSelecting;
      if (!isRange && !isToggle) {
        clearSelection();
        setSelectedSamplePath(sample);
        return;
      }
      e.preventDefault();
      const idx = sampleImages.indexOf(sample);
      if (idx === -1) return;
      setSelectedSet(prev => {
        const anchor = anchorIdxRef.current;
        if (isRange && anchor !== null && anchor < sampleImages.length) {
          const next = new Set(baseSetRef.current);
          const [lo, hi] = anchor < idx ? [anchor, idx] : [idx, anchor];
          for (let i = lo; i <= hi; i++) next.add(sampleImages[i]);
          return next;
        }
        const next = new Set(prev);
        if (isToggle && next.has(sample)) {
          next.delete(sample);
        } else {
          next.add(sample);
        }
        anchorIdxRef.current = idx;
        baseSetRef.current = new Set(next);
        return next;
      });
    },
    [sampleImages, isSelecting, deleteProgress, clearSelection],
  );

  const confirmDeleteSamples = useCallback((paths: string[], all = false) => {
    if (paths.length === 0 || deleteProgress) return;
    openConfirm({
      title: all ? 'Delete All Samples' : 'Delete Selected Samples',
      message: `Delete ${all ? 'all ' : ''}${paths.length} ${all ? '' : 'selected '}sample${paths.length === 1 ? '' : 's'} from "${job.name}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: all ? 'Delete all' : 'Delete selected',
      onConfirm: async () => {
        setDeleteError(null);
        setDeleteProgress({ done: 0, total: paths.length });
        // bounded concurrency so the progress counter advances and the server isn't flooded
        const CONCURRENCY = 4;
        let cursor = 0;
        let done = 0;
        const failed = new Set<string>();
        const worker = async () => {
          while (cursor < paths.length) {
            const imgPath = paths[cursor++];
            try {
              await deleteSample(imgPath);
            } catch (error) {
              failed.add(imgPath);
              console.error('Error deleting sample:', imgPath, error);
            }
            done++;
            setDeleteProgress({ done, total: paths.length });
          }
        };
        await Promise.all(Array.from({ length: Math.min(CONCURRENCY, paths.length) }, worker));
        setDeleteProgress(null);
        setSelectedSet(failed);
        setSelectionMode(failed.size > 0);
        if (failed.size) setDeleteError(`Could not delete ${failed.size} sample${failed.size === 1 ? '' : 's'}. They remain selected so you can retry.`);
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
        await refreshSampleImages();
      },
    });
  }, [deleteProgress, job.name, refreshSampleImages]);

  const deleteSelected = useCallback(() => {
    confirmDeleteSamples(selectedPaths);
  }, [selectedPaths, confirmDeleteSamples]);

  // Delete/Backspace deletes the selection, Escape clears it; ignored while the viewer is open or typing in a field
  useEffect(() => {
    if (!isSelecting || deleteProgress) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (selectedSamplePath) return;
      const target = e.target instanceof HTMLElement ? e.target : null;
      if (target?.closest('input:not([type="checkbox"]), textarea, select, [contenteditable="true"], [role="dialog"]')) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === 'Escape') {
        clearSelection();
        setSelectionMode(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isSelecting, deleteProgress, selectedSamplePath, deleteSelected, clearSelection]);

  const scrollToBottom = () => {
    virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
  };

  const scrollToTop = () => {
    virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start' });
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  const gridCols = rows.reduce((maximum, row) => Math.max(maximum, row.length), 3);

  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      return getSampleConfigFromJob(job);
    }
    return null;
  }, [job]);
  const isPanelLayout = layout === 'panel';

  return (
    <div
      ref={scrollParentCallback}
      className={classNames(
        isPanelLayout
          ? 'relative h-full min-h-[520px] overflow-y-auto'
          : 'absolute top-[108px] left-0 right-0 bottom-0 overflow-y-auto',
      )}
    >
      {sampleImages.length > 0 && (
        <div
          role="toolbar"
          aria-label="Sample actions"
          className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b border-gray-700 bg-gray-900 px-4 py-3 text-sm"
        >
          <span className="mr-auto text-gray-300" role="status">
            {isSelecting
              ? `${selectedPaths.length} of ${sampleImages.length} selected`
              : `${sampleImages.length} sample${sampleImages.length === 1 ? '' : 's'}`}
          </span>
          <Button
            onClick={() => {
              setSelectedSamplePath(null);
              clearSelection();
              setSelectionMode(!isSelecting);
            }}
            disabled={deleteProgress !== null}
            className="rounded px-3 py-2 text-gray-200 hover:bg-gray-700 disabled:opacity-50"
          >
            {isSelecting ? 'Cancel selection' : 'Select samples'}
          </Button>
          {isSelecting && (
            <>
              <Button
                onClick={() => {
                  setSelectedSet(new Set(sampleImages));
                  anchorIdxRef.current = null;
                  baseSetRef.current = new Set();
                }}
                disabled={deleteProgress !== null || selectedPaths.length === sampleImages.length}
                className="rounded px-3 py-2 text-gray-200 hover:bg-gray-700 disabled:opacity-50"
              >
                Select all
              </Button>
              <Button
                onClick={clearSelection}
                disabled={deleteProgress !== null || selectedPaths.length === 0}
                className="rounded px-3 py-2 text-gray-200 hover:bg-gray-700 disabled:opacity-50"
              >
                Clear selection
              </Button>
              <Button
                onClick={deleteSelected}
                disabled={deleteProgress !== null || selectedPaths.length === 0}
                className="inline-flex items-center gap-2 rounded px-3 py-2 text-red-300 hover:bg-red-950 disabled:opacity-50"
              >
                <LuTrash2 aria-hidden="true" />
                Delete selected
              </Button>
              {job.job_type === 'generate' && <Button
                className="rounded px-3 py-2 text-blue-300 hover:bg-gray-700 disabled:opacity-50"
                disabled={!selectedPaths.length || preservationBusy}
                onClick={() => { preservationRequest.current = ''; setPreservationOpen(true); setPreservationMessage(''); }}>
                Use selected samples for preservation
              </Button>}
            </>
          )}
          <Button
            onClick={() => confirmDeleteSamples([...sampleImages], true)}
            disabled={deleteProgress !== null}
            className="inline-flex items-center gap-2 rounded px-3 py-2 text-red-300 hover:bg-red-950 disabled:opacity-50"
          >
            <LuTrash2 aria-hidden="true" />
            Delete all
          </Button>
          {deleteError && (
            <div role="alert" className="flex w-full items-center gap-3 rounded border border-red-900 bg-red-950/40 p-3 text-red-200">
              <span className="flex-1">{deleteError}</span>
              <Button onClick={() => setDeleteError(null)} className="rounded px-2 py-1 hover:bg-red-900">
                Dismiss
              </Button>
            </div>
          )}
        </div>
      )}
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && rows.length > 0 && scrollParent && (
          <Virtuoso
            ref={virtuosoRef}
            customScrollParent={scrollParent}
            totalCount={rows.length}
            initialTopMostItemIndex={rows.length - 1}
            followOutput="auto"
            increaseViewportBy={400}
            computeItemKey={index => rows[index]?.[0] ?? index}
            itemContent={index => {
              const row = rows[index];
              if (!row) return null;

              // Only pad the final row when numSamples < MIN_COLS and the row is short.
              const MIN_COLS = 3;
              const shouldPad = numSamples < MIN_COLS && row.length < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - row.length : 0;

              return (
                // pb-1 recreates the vertical gap between rows that the original single CSS grid provided via `gap-1`.
                <div className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}>
                  {row.map(sample => (
                    <SampleImageCard
                      key={sample}
                      imageUrl={sample}
                      numSamples={numSamples}
                      sampleImages={sampleImages}
                      alt="Sample Image"
                      onClick={e => handleCardClick(sample, e)}
                      selected={selectedSet.has(sample) || selectedSamplePath === sample}
                      selectionChecked={selectedSet.has(sample)}
                      onSelect={isSelecting ? checked => setSampleSelected(sample, checked) : undefined}
                      selectionLabel={`Select sample ${sampleImages.indexOf(sample) + 1}`}
                      observerRoot={scrollParent}
                    />
                  ))}
                  {Array.from({ length: padsNeeded }).map((_, i) => (
                    <div key={`pad-${index}-${i}`} className="invisible" />
                  ))}
                </div>
              );
            }}
          />
        )}
      </div>
      {preservationMessage && <p role="status" className="p-3 text-sm text-blue-300">{preservationMessage}</p>}
      <Dialog open={preservationOpen} onClose={() => { if (!preservationBusy) setPreservationOpen(false); }} className="relative z-30">
        <DialogBackdrop className="fixed inset-0 bg-gray-900/75" />
        <div className="fixed inset-0 flex items-center justify-center p-4">
          <DialogPanel className="w-full max-w-md rounded-lg bg-white p-6 text-gray-900 dark:bg-gray-900 dark:text-gray-200">
            <DialogTitle className="font-semibold">Create preservation dataset</DialogTitle>
            <p className="my-3 text-sm">Use these {selectedPaths.length} reviewed images and their recorded prompts to preserve the source LoRA. Keep separate prompts and seeds for evaluation.</p>
            <label className="block text-sm">Dataset name<input aria-label="Preservation dataset name" className="my-2 w-full rounded border p-2 dark:border-gray-700 dark:bg-gray-950" value={preservationName} onChange={event => { setPreservationName(event.target.value); preservationRequest.current = ''; }} disabled={preservationBusy} /></label>
            {preservationMessage && <p role="alert" className="my-2 text-sm text-red-400">{preservationMessage}</p>}
            <div className="mt-4 flex justify-end gap-3">
              <Button disabled={preservationBusy} onClick={() => setPreservationOpen(false)}>Cancel</Button>
              <Button className="rounded bg-blue-600 px-3 py-2 text-white disabled:opacity-50" disabled={preservationBusy || !preservationName.trim()} onClick={() => void createPreservation()}>{preservationBusy ? 'Creating…' : 'Create dataset'}</Button>
            </div>
          </DialogPanel>
        </div>
      </Dialog>
      <Dialog open={deleteProgress !== null} onClose={() => {}} className="relative z-20">
        <DialogBackdrop className="fixed inset-0 bg-gray-900/75" />
        <div className="fixed inset-0 z-10 flex items-center justify-center p-4">
          <DialogPanel className="w-full max-w-sm rounded-lg bg-gray-800 p-6 shadow-xl text-gray-200">
            <DialogTitle as="h3" className="text-base font-semibold flex items-center gap-2">
              <LuLoader className="animate-spin" />
              Deleting Samples
            </DialogTitle>
            <p className="mt-2 text-sm text-gray-400">
              {deleteProgress?.done ?? 0} / {deleteProgress?.total ?? 0}
            </p>
            <div className="mt-3 h-2 w-full rounded bg-gray-700 overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-150"
                style={{
                  width: `${deleteProgress && deleteProgress.total > 0 ? (deleteProgress.done / deleteProgress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </DialogPanel>
        </div>
      </Dialog>
      <SampleImageViewer
        sampleMetadata={sampleMetadata}
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setSelectedSamplePath}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className={classNames(
          'z-10 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer',
          isPanelLayout ? 'absolute bottom-16 right-4' : 'fixed bottom-[72px] right-6',
        )}
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className={classNames(
          'z-10 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer',
          isPanelLayout ? 'absolute bottom-4 right-4' : 'fixed bottom-5 right-6',
        )}
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
