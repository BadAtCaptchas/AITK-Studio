'use client';
import { useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { openConfirm } from '@/components/ConfirmModal';
import type { JobConfig } from '@/types';

export type TrainingDraft = { config: JobConfig; gpuIDs: string | null; workerID: string };
type StoredDraft = { draft: TrainingDraft; baseline: string };
const drafts = new Map<string, StoredDraft>();
let returnPath = '/jobs/new';
export const getTrainingReturnPath = () => returnPath;
export const readTrainingDraft = (key: string) => drafts.get(key)?.draft;
const serialize = (draft: TrainingDraft) => JSON.stringify(draft);
type NavigateEvent = Event & { canIntercept: boolean; hashChange: boolean; destination: { url: string }; navigationType: string };

export function useTrainingDraft(key: string, draft: TrainingDraft) {
  const router = useRouter();
  const current = useRef(draft); current.current = draft;
  const baseline = useRef(drafts.get(key)?.baseline || serialize(draft));
  const interacted = useRef(!!drafts.get(key));
  const dirty = useRef(false);
  const allowing = useRef(false), confirming = useRef(false);
  const currentURL = useRef('');
  const currentState = useRef<unknown>(null);
  useEffect(() => {
    if (!interacted.current) baseline.current = serialize(draft);
    dirty.current = serialize(draft) !== baseline.current;
  }, [draft.config, draft.workerID, draft.gpuIDs]);
  const leaveSetup = useCallback((onLeave: () => void) => {
    if (allowing.current || !dirty.current) { onLeave(); return; }
    if (confirming.current) return;
    drafts.set(key, { draft: current.current, baseline: baseline.current });
    confirming.current = true;
    openConfirm({
      title: 'Leave training setup?',
      message: 'Your changes have not been saved as a run. Your draft stays available in this tab until it is reloaded.',
      confirmText: 'Leave setup',
      onCancel: () => { confirming.current = false; },
      onConfirm: () => { confirming.current = false; allowing.current = true; onLeave(); },
    });
  }, [key]);
  useEffect(() => {
    returnPath = window.location.pathname + window.location.search;
    currentURL.current = window.location.href; currentState.current = window.history.state;
    const sameDocument = (destination: string) => {
      const url = new URL(destination, window.location.href), here = new URL(currentURL.current);
      return url.origin === here.origin && url.pathname === here.pathname && url.search === here.search;
    };
    const go = (href: string) => {
      const url = new URL(href, window.location.href);
      if (url.origin === window.location.origin) router.push(url.pathname + url.search + url.hash);
      else window.location.assign(url.href);
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (dirty.current && !allowing.current) { event.preventDefault(); event.returnValue = ''; }
    };
    const navigate = (event: MouseEvent) => {
      if (!dirty.current || allowing.current || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const anchor = event.target instanceof Element ? event.target.closest('a') : null;
      if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download') || sameDocument(anchor.href)) return;
      event.preventDefault(); event.stopPropagation();
      leaveSetup(() => go(anchor.href));
    };
    // Navigation API runs before same-document history traversal, including browser Back/Forward.
    const navigation = (window as Window & { navigation?: EventTarget }).navigation;
    const browserNavigate = (event: Event) => {
      const next = event as NavigateEvent;
      if (!dirty.current || allowing.current || !next.canIntercept || next.hashChange || sameDocument(next.destination.url)) return;
      event.preventDefault();
      leaveSetup(() => go(next.destination.url));
    };
    const popstate = (event: PopStateEvent) => {
      if (navigation || allowing.current || !dirty.current || sameDocument(window.location.href)) return;
      const destination = window.location.href;
      event.stopImmediatePropagation();
      window.history.replaceState(currentState.current, '', currentURL.current);
      leaveSetup(() => go(destination));
    };
    document.addEventListener('click', navigate, true);
    navigation?.addEventListener('navigate', browserNavigate);
    window.addEventListener('popstate', popstate, true);
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      if (dirty.current) drafts.set(key, { draft: current.current, baseline: baseline.current });
      document.removeEventListener('click', navigate, true);
      navigation?.removeEventListener('navigate', browserNavigate);
      window.removeEventListener('popstate', popstate, true);
      window.removeEventListener('beforeunload', beforeUnload);
    };
  }, [key, router, leaveSetup]);
  const noteFormInteraction = () => {
    if (!interacted.current) baseline.current = serialize(current.current);
    interacted.current = true;
  };
  return {
    leaveSetup,
    formTracking: { onPointerDownCapture: noteFormInteraction, onKeyDownCapture: noteFormInteraction, onChangeCapture: noteFormInteraction },
    markSaved: () => { baseline.current = serialize(current.current); dirty.current = false; drafts.delete(key); allowing.current = true; },
  };
}
