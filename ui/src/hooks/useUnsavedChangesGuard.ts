'use client';

import { useEffect, useRef } from 'react';
import { usePathname, useRouter } from '@/navigation';
import { openConfirm } from '@/components/ConfirmModal';
import { registerHistoryGuard, type HistoryTransition } from '@/components/HistoryNavigationGuard';
import useCommittedRef from './useCommittedRef';

type GuardOptions = {
  dirty: boolean;
  revision: string;
  title: string;
  message: string;
  confirmText?: string;
};

type GuardNavigation = EventTarget & { traverseTo?: (key: string) => unknown };
type GuardNavigateEvent = Event & {
  canIntercept: boolean;
  hashChange: boolean;
  navigationType: string;
  destination: { url: string; key?: string };
};

/** Protect a page-local draft without persisting its contents. Query/hash changes
 * within the current workspace stay usable (for example retained monitor tabs). */
export default function useUnsavedChangesGuard(options: GuardOptions): void {
  const current = useCommittedRef(options);
  const pathname = usePathname();
  const router = useRouter();
  const approved = useRef<{ destination: string; revision: string } | null>(null);

  useEffect(() => {
    let alive = true;
    let confirming = false;
    const navigation = (window as Window & { navigation?: GuardNavigation }).navigation;
    const sameWorkspace = (href: string) => {
      const target = new URL(href, window.location.href);
      return target.origin === window.location.origin && target.pathname === pathname;
    };
    const isApproved = (destination: string) =>
      approved.current?.destination === destination && approved.current.revision === current.current.revision;
    const go = (href: string) => {
      const target = new URL(href, window.location.href);
      if (target.origin === window.location.origin) router.push(target.pathname + target.search + target.hash);
      else window.location.assign(target.href);
    };
    const confirmLeave = (destination: string, resume: () => void, cancel?: () => void) => {
      if (confirming) return;
      confirming = true;
      openConfirm({
        title: current.current.title,
        message: current.current.message,
        confirmText: current.current.confirmText ?? 'Discard changes',
        type: 'warning',
        onCancel: () => {
          confirming = false;
          cancel?.();
        },
        onConfirm: () => {
          confirming = false;
          if (!alive) return;
          approved.current = { destination, revision: current.current.revision };
          resume();
        },
      });
    };
    const click = (event: MouseEvent) => {
      if (
        !current.current.dirty ||
        event.defaultPrevented ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey
      )
        return;
      const anchor = event.target instanceof Element ? event.target.closest('a') : null;
      if (
        !anchor ||
        (anchor.target && anchor.target !== '_self') ||
        anchor.hasAttribute('download') ||
        !['http:', 'https:'].includes(anchor.protocol) ||
        sameWorkspace(anchor.href) ||
        isApproved(anchor.href)
      )
        return;
      event.preventDefault();
      event.stopPropagation();
      confirmLeave(anchor.href, () => go(anchor.href));
    };
    const navigate = (event: Event) => {
      const next = event as GuardNavigateEvent;
      if (
        !current.current.dirty ||
        !event.cancelable ||
        !next.canIntercept ||
        next.hashChange ||
        sameWorkspace(next.destination.url) ||
        isApproved(next.destination.url)
      )
        return;
      event.preventDefault();
      confirmLeave(next.destination.url, () => {
        if (next.navigationType === 'traverse' && next.destination.key && navigation?.traverseTo)
          navigation.traverseTo(next.destination.key);
        else go(next.destination.url);
      });
    };
    const popstate = ({ destination, resume, cancel }: HistoryTransition) => {
      if (navigation || !current.current.dirty || sameWorkspace(destination) || isApproved(destination)) return false;
      if (confirming) {
        cancel();
        return true;
      }
      confirmLeave(destination, resume, cancel);
      return true;
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!current.current.dirty || approved.current?.revision === current.current.revision) return;
      event.preventDefault();
      event.returnValue = '';
    };
    document.addEventListener('click', click, true);
    navigation?.addEventListener('navigate', navigate);
    const unregisterHistoryGuard = registerHistoryGuard(popstate);
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      alive = false;
      approved.current = null;
      document.removeEventListener('click', click, true);
      navigation?.removeEventListener('navigate', navigate);
      unregisterHistoryGuard();
      window.removeEventListener('beforeunload', beforeUnload);
    };
  }, [pathname, router, current]);
}
