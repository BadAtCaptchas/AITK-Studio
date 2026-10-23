'use client';
import useCommittedRef from './useCommittedRef';
import { useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { openConfirm } from '@/components/ConfirmModal';
import { registerHistoryGuard, type HistoryTransition } from '@/components/HistoryNavigationGuard';
import { saveStoredTrainingDraft, removeStoredTrainingDraft, type StoredTrainingDraft, type TrainingDraft } from '@/utils/trainingDraftStorage';
export type { TrainingDraft } from '@/utils/trainingDraftStorage';
type StoredDraft = { draft: TrainingDraft; baseline: string };
const drafts = new Map<string, StoredDraft>();
let returnPath = '/jobs/new';
export const getTrainingReturnPath = () => returnPath;
export const readTrainingDraft = (key: string) => drafts.get(key)?.draft;
export const restoreTrainingDraft = (key: string, value: StoredTrainingDraft) => drafts.set(key, value);
const serialize = (draft: TrainingDraft) => JSON.stringify(draft);
type NavigateEvent = Event & {
  canIntercept: boolean;
  hashChange: boolean;
  destination: { url: string; key?: string };
  navigationType: string;
};

export function useTrainingDraft(key: string, draft: TrainingDraft, hasUnsavedConfigText = false) {
  const router = useRouter();
  const current = useCommittedRef(draft);
  const baseline = useRef(drafts.get(key)?.baseline || serialize(draft));
  const interacted = useRef(!!drafts.get(key));
  const dirty = useRef(false);
  const allowing = useRef(false),
    confirming = useRef(false);
  const currentURL = useRef('');
  useEffect(() => {
    if (allowing.current) {
      dirty.current = false;
      return;
    }
    if (!interacted.current) baseline.current = serialize(draft);
    dirty.current = hasUnsavedConfigText || serialize(draft) !== baseline.current;
    if (!dirty.current) {
      // An earlier autosave or canceled leave may have captured values that the
      // user has since reverted. Do not resurrect them on the next visit.
      drafts.delete(key);
      removeStoredTrainingDraft(key);
      return;
    }
    const timer = window.setTimeout(() => {
      if (dirty.current && !allowing.current) saveStoredTrainingDraft(key, { draft: current.current, baseline: baseline.current });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [draft.config, draft.workerID, draft.gpuIDs, draft.migration, hasUnsavedConfigText, key, current]);
  const leaveSetup = useCallback(
    (onLeave: () => void, onCancel?: () => void) => {
      if (allowing.current || !dirty.current) {
        onLeave();
        return;
      }
      if (confirming.current) return;
      drafts.set(key, { draft: current.current, baseline: baseline.current });
      saveStoredTrainingDraft(key, { draft: current.current, baseline: baseline.current });
      confirming.current = true;
      openConfirm({
        title: 'Leave training setup?',
        message:
          'Your changes have not been saved as a run. Your draft stays available in this tab. Parsed settings can be recovered after reload; unfinished Config text stays in memory only.',
        confirmText: 'Leave setup',
        onCancel: () => {
          confirming.current = false;
          onCancel?.();
        },
        onConfirm: () => {
          confirming.current = false;
          allowing.current = true;
          onLeave();
        },
      });
    },
    [key],
  );
  useEffect(() => {
    returnPath = window.location.pathname + window.location.search;
    currentURL.current = window.location.href;
    const sameDocument = (destination: string) => {
      const url = new URL(destination, window.location.href),
        here = new URL(currentURL.current);
      return url.origin === here.origin && url.pathname === here.pathname && url.search === here.search;
    };
    const go = (href: string) => {
      const url = new URL(href, window.location.href);
      if (url.origin === window.location.origin) router.push(url.pathname + url.search + url.hash);
      else window.location.assign(url.href);
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (dirty.current && !allowing.current) {
        saveStoredTrainingDraft(key, { draft: current.current, baseline: baseline.current });
        event.preventDefault();
        event.returnValue = '';
      }
    };
    const navigate = (event: MouseEvent) => {
      if (
        !dirty.current ||
        allowing.current ||
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      )
        return;
      const anchor = event.target instanceof Element ? event.target.closest('a') : null;
      if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download') || sameDocument(anchor.href)) return;
      event.preventDefault();
      event.stopPropagation();
      leaveSetup(() => go(anchor.href));
    };
    // Navigation API runs before same-document history traversal, including browser Back/Forward.
    const navigation = (window as Window & { navigation?: EventTarget & { traverseTo?: (key: string) => unknown } }).navigation;
    const browserNavigate = (event: Event) => {
      const next = event as NavigateEvent;
      if (
        !dirty.current ||
        allowing.current ||
        !next.canIntercept ||
        next.hashChange ||
        sameDocument(next.destination.url)
      )
        return;
      event.preventDefault();
      leaveSetup(() => {
        if (next.navigationType === 'traverse' && next.destination.key && navigation?.traverseTo)
          navigation.traverseTo(next.destination.key);
        else go(next.destination.url);
      });
    };
    const popstate = ({ destination, resume, cancel }: HistoryTransition) => {
      if (navigation || allowing.current || !dirty.current || sameDocument(destination)) return false;
      if (confirming.current) { cancel(); return true; }
      leaveSetup(resume, cancel);
      return true;
    };
    document.addEventListener('click', navigate, true);
    navigation?.addEventListener('navigate', browserNavigate);
    const unregisterHistoryGuard = registerHistoryGuard(popstate);
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      if (dirty.current) {
        drafts.set(key, { draft: current.current, baseline: baseline.current });
        saveStoredTrainingDraft(key, { draft: current.current, baseline: baseline.current });
      }
      document.removeEventListener('click', navigate, true);
      navigation?.removeEventListener('navigate', browserNavigate);
      unregisterHistoryGuard();
      window.removeEventListener('beforeunload', beforeUnload);
    };
  }, [key, router, leaveSetup]);
  const noteFormInteraction = () => {
    if (!interacted.current) baseline.current = serialize(current.current);
    interacted.current = true;
  };
  return {
    leaveSetup,
    formTracking: {
      onPointerDownCapture: noteFormInteraction,
      onKeyDownCapture: noteFormInteraction,
      onChangeCapture: noteFormInteraction,
    },
    markSaved: () => {
      baseline.current = serialize(current.current);
      dirty.current = false;
      drafts.delete(key);
      removeStoredTrainingDraft(key);
      allowing.current = true;
    },
  };
}
