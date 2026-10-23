'use client';
import useScopedState from '@/hooks/useScopedState';
import useUnsavedChangesGuard from '@/hooks/useUnsavedChangesGuard';
import { useEffect, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';

export default function JobNotes({ job }: { job: Job }) {
  const [notes, setNotes] = useScopedState(job.id, '');
  const [savedNotes, setSavedNotes] = useScopedState(job.id, '');
  const [loaded, setLoaded] = useScopedState(job.id, false);
  const [status, setStatus] = useState('Loading…');
  useUnsavedChangesGuard({
    dirty: loaded && notes !== savedNotes,
    revision: notes,
    title: 'Leave notes?',
    message: status === 'Saving…'
      ? 'Your notes are still saving. Leaving will not cancel the save already requested.'
      : 'Your unsaved notes will be discarded if you leave this job.',
  });
  useEffect(() => {
    const controller = new AbortController();
    void apiClient
      .get<{ notes?: unknown }>(`/api/jobs/${job.id}/notes`, { signal: controller.signal })
      .then(({ data }) => {
        if (typeof data.notes !== 'string') throw new Error('Invalid notes response');
        if (!controller.signal.aborted) {
          setNotes(data.notes);
          setSavedNotes(data.notes);
          setLoaded(true);
          setStatus('');
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setStatus('Could not load notes.');
      });
    return () => controller.abort();
  }, [job.id, setNotes, setSavedNotes, setLoaded]);
  return (
    <div className="space-y-3 p-5">
      <label htmlFor="job-notes" className="block font-semibold">
        Job notes
      </label>
      <textarea
        id="job-notes"
        disabled={!loaded || status === 'Saving…'}
        value={notes}
        onChange={e => {
          setNotes(e.target.value);
          setStatus(e.target.value === savedNotes ? '' : 'Unsaved');
        }}
        className="w-full min-h-[50vh] rounded border border-gray-700 bg-gray-950 p-4 font-mono text-sm"
      />
      <div className="flex items-center gap-4">
        <button
          className="operator-button"
          disabled={!loaded || status === 'Saving…'}
          onClick={async () => {
            setStatus('Saving…');
            try {
              await apiClient.post(`/api/jobs/${job.id}/notes`, { notes });
              setSavedNotes(notes);
              setStatus('Saved');
            } catch {
              setStatus('Could not save notes.');
            }
          }}
        >
          Save notes
        </button>
        <span role="status" className="text-sm text-gray-400">
          {status}
        </span>
      </div>
    </div>
  );
}
