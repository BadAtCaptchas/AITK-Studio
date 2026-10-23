'use client';

import { useLayoutEffect } from 'react';

export type HistoryTransition = { destination: string; resume: () => void; cancel: () => void };
type HistoryGuard = (transition: HistoryTransition) => boolean;
const guards = new Set<HistoryGuard>();
const removed = new Set<(guard: HistoryGuard) => void>();
const markerKey = '__aitk_history_entry';
const documentKey = () => window.crypto?.randomUUID?.() ?? `history-${Date.now()}-${Math.random()}`;
type Marker = { document: string; index: number };
type Entry = { href: string; state: unknown; index: number };
type Pending = {
  source: Entry;
  target: Entry;
  guard: HistoryGuard;
  phase: 'restoring' | 'waiting' | 'resuming';
  decision: 'pending' | 'cancel' | 'resume';
};

function marker(state: unknown): Marker | null {
  if (!state || typeof state !== 'object' || !(markerKey in state)) return null;
  const value = state[markerKey];
  return value &&
    typeof value === 'object' &&
    'document' in value &&
    typeof value.document === 'string' &&
    'index' in value &&
    typeof value.index === 'number' &&
    Number.isSafeInteger(value.index)
    ? { document: value.document, index: value.index }
    : null;
}

function markedState(state: unknown, value: Marker) {
  return { ...(state && typeof state === 'object' ? state : {}), [markerKey]: value };
}

function sameDocument(left: string, right: string): boolean {
  const a = new URL(left),
    b = new URL(right);
  return a.origin === b.origin && a.pathname === b.pathname && a.search === b.search;
}

export function registerHistoryGuard(guard: HistoryGuard): () => void {
  guards.add(guard);
  return () => {
    guards.delete(guard);
    removed.forEach(listener => listener(guard));
  };
}

/** Install before the router's passive history listener. Registering a listener
 * only when an editor mounts is too late: the router can unmount it first. */
export default function HistoryNavigationGuard() {
  useLayoutEffect(() => {
    if ((window as Window & { navigation?: unknown }).navigation) return;
    const history = window.history;
    const originalPush = history.pushState,
      originalReplace = history.replaceState;
    let active = true;
    let documentID = marker(history.state)?.document ?? documentKey();
    let current: Entry = { href: window.location.href, state: history.state, index: marker(history.state)?.index ?? 0 };
    let pending: Pending | null = null;
    const stamp = (index: number, state: unknown = history.state): Entry => {
      originalReplace.call(history, markedState(state, { document: documentID, index }), '', window.location.href);
      return { href: window.location.href, state: history.state, index };
    };
    current = stamp(current.index);
    const push: History['pushState'] = (state: unknown, unused: string, url?: string | URL | null) => {
      if (!active) return originalPush.call(history, state, unused, url);
      const index = current.index + 1;
      originalPush.call(history, markedState(state, { document: documentID, index }), unused, url);
      pending = null;
      current = { href: window.location.href, state: history.state, index };
    };
    const replace: History['replaceState'] = (state: unknown, unused: string, url?: string | URL | null) => {
      if (!active) return originalReplace.call(history, state, unused, url);
      const existing = marker(history.state);
      const index = existing?.document === documentID ? existing.index : current.index;
      originalReplace.call(history, markedState(state, { document: documentID, index }), unused, url);
      current = { href: window.location.href, state: history.state, index };
    };
    history.pushState = push;
    history.replaceState = replace;
    const resume = (attempt: Pending) => {
      if (pending !== attempt || !active || !guards.has(attempt.guard)) return;
      attempt.decision = 'resume';
      if (attempt.phase !== 'waiting') return;
      attempt.phase = 'resuming';
      history.go(attempt.target.index - attempt.source.index);
    };
    const cancel = (attempt: Pending) => {
      if (pending !== attempt) return;
      attempt.decision = 'cancel';
      if (attempt.phase === 'waiting') pending = null;
    };
    const unregister = (guard: HistoryGuard) => {
      if (pending?.guard === guard) cancel(pending);
    };
    removed.add(unregister);
    const popstate = (event: PopStateEvent) => {
      let nextMarker = marker(history.state);
      let target: Entry;
      if (
        (!nextMarker || (nextMarker.document === documentID && nextMarker.index === current.index)) &&
        window.location.href !== current.href &&
        sameDocument(window.location.href, current.href)
      ) {
        // A native hash anchor can create an entry without calling pushState.
        target = stamp(current.index + 1, history.state ?? current.state);
        nextMarker = marker(target.state);
      } else if (nextMarker?.document === documentID) {
        target = { href: window.location.href, state: history.state, index: nextMarker.index };
      } else {
        // Entries predating this document's tracking cannot be assigned a safe
        // traversal delta. Never replace a foreign entry with the editor URL.
        pending = null;
        documentID = documentKey();
        current = stamp(0);
        return;
      }
      const attempt = pending;
      if (attempt) {
        if (attempt.phase === 'resuming' && target.index === attempt.target.index) {
          pending = null;
          current = target;
          return;
        }
        event.stopImmediatePropagation();
        if (target.index !== attempt.source.index) {
          attempt.phase = 'restoring';
          history.go(attempt.source.index - target.index);
          return;
        }
        current = target;
        attempt.phase = 'waiting';
        if (attempt.decision === 'resume') resume(attempt);
        else if (attempt.decision === 'cancel') pending = null;
        return;
      }
      if (target.index !== current.index) {
        for (const guard of guards) {
          const next: Pending = { source: current, target, guard, phase: 'restoring', decision: 'pending' };
          pending = next;
          if (guard({ destination: target.href, resume: () => resume(next), cancel: () => cancel(next) })) {
            event.stopImmediatePropagation();
            history.go(current.index - target.index);
            return;
          }
          pending = null;
        }
      }
      current = target;
    };
    const hashchange = () => {
      if (!pending && window.location.href !== current.href && sameDocument(window.location.href, current.href))
        current = stamp(current.index + 1, history.state ?? current.state);
    };
    window.addEventListener('popstate', popstate, true);
    window.addEventListener('hashchange', hashchange, true);
    return () => {
      active = false;
      pending = null;
      removed.delete(unregister);
      window.removeEventListener('popstate', popstate, true);
      window.removeEventListener('hashchange', hashchange, true);
      // Next may wrap our functions later. Leave that wrapper intact; captured
      // functions become inert so its own cleanup remains safe as well.
      if (history.pushState === push) history.pushState = originalPush;
      if (history.replaceState === replace) history.replaceState = originalReplace;
    };
  }, []);
  return null;
}
