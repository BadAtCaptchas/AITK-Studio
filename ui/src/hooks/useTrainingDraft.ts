'use client';
import { useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { openConfirm } from '@/components/ConfirmModal';
import type { JobConfig } from '@/types';

export type TrainingDraft = { config: JobConfig; gpuIDs: string | null; workerID: string };
// Memory only: drafts may contain sensitive configuration. Never put them in browser storage.
const drafts = new Map<string, TrainingDraft>();
let returnPath = '/jobs/new';
export const getTrainingReturnPath = () => returnPath;
export const readTrainingDraft = (key: string) => drafts.get(key);

export function useTrainingDraft(key: string, draft: TrainingDraft) {
  const router = useRouter();
  const current = useRef(draft);
  const dirty = useRef(Boolean(drafts.get(key)));
  const interacted = useRef(false);
  const previous = useRef(draft);
  current.current = draft;
  useEffect(() => {
    if (
      interacted.current &&
      (previous.current.config !== draft.config ||
        previous.current.workerID !== draft.workerID ||
        previous.current.gpuIDs !== draft.gpuIDs)
    )
      dirty.current = true;
    previous.current = draft;
  }, [draft.config, draft.workerID, draft.gpuIDs]);
  useEffect(() => {
    returnPath = window.location.pathname + window.location.search;
    const noteInteraction = () => {
      interacted.current = true;
    };
    const markDirty = () => {
      dirty.current = true;
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (dirty.current) {
        event.preventDefault();
        event.returnValue = '';
      }
    };
    const navigate = (event: MouseEvent) => {
      if (
        !dirty.current ||
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      )
        return;
      const anchor = event.target instanceof Element ? event.target.closest('a') : null;
      if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download')) return;
      const url = new URL(anchor.href, window.location.href);
      if (url.pathname === window.location.pathname || url.searchParams.get('returnTo') === 'training') return;
      event.preventDefault();
      event.stopPropagation();
      drafts.set(key, current.current);
      openConfirm({
        title: 'Leave training setup?',
        message:
          'Your changes have not been saved as a run. Your draft stays available in this tab until it is reloaded.',
        confirmText: 'Leave setup',
        onConfirm: () => {
          if (url.origin === window.location.origin) router.push(`${url.pathname}${url.search}${url.hash}`);
          else window.location.assign(url.href);
        },
      });
    };
    document.addEventListener('pointerdown', noteInteraction, true);
    document.addEventListener('keydown', noteInteraction, true);
    document.addEventListener('input', markDirty);
    document.addEventListener('change', markDirty);
    document.addEventListener('click', navigate, true);
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      if (dirty.current) drafts.set(key, current.current);
      document.removeEventListener('pointerdown', noteInteraction, true);
      document.removeEventListener('keydown', noteInteraction, true);
      document.removeEventListener('input', markDirty);
      document.removeEventListener('change', markDirty);
      document.removeEventListener('click', navigate, true);
      window.removeEventListener('beforeunload', beforeUnload);
    };
  }, [key, router]);
  return {
    leaveSetup: (onLeave: () => void) => {
      if (!dirty.current) {
        onLeave();
        return;
      }
      drafts.set(key, current.current);
      openConfirm({
        title: 'Leave training setup?',
        message:
          'Your changes have not been saved as a run. Your draft stays available in this tab until it is reloaded.',
        confirmText: 'Leave setup',
        onConfirm: onLeave,
      });
    },
    markSaved: () => {
      dirty.current = false;
      drafts.delete(key);
    },
  };
}
