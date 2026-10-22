'use client';
import { useEffect, useRef, useState, type RefObject } from 'react';
import { Box, Database, FileText, Image, Search, Settings2, Target } from 'lucide-react';
import type { JobConfig } from '@/types';
import type { TrainingFieldTarget } from '@/utils/trainingValidation';
import styles from './trainingWorkspace.module.css';

const sections = [
  { id: 'job-basics', title: 'Basics', Icon: FileText },
  { id: 'job-model', title: 'Model', Icon: Box },
  { id: 'job-training', title: 'Training', Icon: Settings2 },
  { id: 'job-dataset', title: 'Dataset', Icon: Database },
  { id: 'job-samples', title: 'Samples', Icon: Image },
  { id: 'job-validation', title: 'Validation', Icon: Target },
] as const;

function revealElement(wrapper: HTMLElement): void {
  let ancestor: HTMLElement | null = wrapper;
  while (ancestor) {
    if (ancestor instanceof HTMLDetailsElement) ancestor.open = true;
    ancestor = ancestor.parentElement;
  }
  const control = wrapper.querySelector<HTMLElement>(
    'input:not([type="file"]):not(:disabled), select:not(:disabled), textarea:not(:disabled), [role="switch"], [role="combobox"], button:not(:disabled)',
  ) ?? wrapper.querySelector<HTMLElement>('h2') ?? wrapper;
  if (!control.matches('input,select,textarea,button,[tabindex]')) control.tabIndex = -1;
  control.focus({ preventScroll: true });
  control.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

export function revealTrainingField(form: HTMLElement, target: TrainingFieldTarget): void {
  const matches = Array.from(form.querySelectorAll<HTMLElement>('[data-field-label]'))
    .filter(element => element.dataset.fieldLabel === target.label);
  const wrapper = matches[target.index ?? 0] ?? form.querySelector<HTMLElement>(`#${CSS.escape(target.step)}`);
  if (wrapper) revealElement(wrapper);
}

export default function TrainingSectionNav({ formRef, config }: { formRef: RefObject<HTMLFormElement | null>; config: JobConfig }) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState<string>('job-basics');
  const [matches, setMatches] = useState<Array<{ label: string; section: string; element: HTMLElement }>>([]);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sync = () => {
      const hash = window.location.hash.slice(1);
      if (sections.some(section => section.id === hash)) setActive(hash);
    };
    sync();
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, []);

  useEffect(() => {
    const root = formRef.current?.closest<HTMLElement>('[role="tabpanel"]');
    if (!root) return;
    let frame = 0;
    const syncScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const top = root.getBoundingClientRect().top + (window.innerWidth < 768 ? 120 : 24);
        let current: string = sections[0].id;
        for (const section of sections) {
          const element = formRef.current?.querySelector<HTMLElement>(`#${section.id}`);
          if (element && element.getBoundingClientRect().top <= top) current = section.id;
        }
        setActive(current);
      });
    };
    root.addEventListener('scroll', syncScroll, { passive: true });
    window.addEventListener('resize', syncScroll);
    syncScroll();
    return () => { cancelAnimationFrame(frame); root.removeEventListener('scroll', syncScroll); window.removeEventListener('resize', syncScroll); };
  }, [formRef]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
    if (!query.trim()) { setMatches([]); return; }
    const terms = query.toLowerCase().trim().split(/\s+/);
    const found = Array.from(formRef.current?.querySelectorAll<HTMLElement>('[data-field-label]') ?? [])
      .filter(element => {
        const label = element.dataset.fieldLabel?.toLowerCase() ?? '';
        return terms.every(term => label.includes(term));
      })
      .map(element => ({
        element,
        label: element.dataset.fieldLabel ?? '',
        section: sections.find(section => section.id === element.closest('fieldset[id]')?.id)?.title ?? 'Model',
      }));
    setMatches(found.slice(0, 30));
    });
    return () => cancelAnimationFrame(frame);
  }, [query, config, formRef]);

  return (
    <nav className={styles.navigator} aria-label="Training sections">
      <div className={styles.finder}>
        <Search className="h-3.5 w-3.5 flex-none text-gray-400" aria-hidden="true" />
        <input type="search" aria-label="Find parameter" placeholder="Find parameter..." value={query}
          onChange={event => setQuery(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'ArrowDown') { event.preventDefault(); resultsRef.current?.querySelector('button')?.focus(); }
            if (event.key === 'Enter') { event.preventDefault(); if (matches[0]) revealElement(matches[0].element); }
            if (event.key === 'Escape') setQuery('');
          }} />
      </div>
      {query.trim() ? (
        <div className={styles.searchResults} ref={resultsRef} aria-label="Matching parameters">
          <p className="px-3 py-2 text-xs text-gray-500" role="status">{matches.length ? `${matches.length} matching parameters` : 'No applicable parameters'}</p>
          {matches.map((match, index) => (
            <button type="button" key={`${match.label}-${index}`} className={styles.searchResult}
              onClick={() => revealElement(match.element)}>
              <span>{match.label}</span><small>{match.section}</small>
            </button>
          ))}
        </div>
      ) : (
        <div className={styles.sectionLinks}>
          {sections.map(({ id, title, Icon }) => (
            <a key={id} href={`#${id}`} aria-current={active === id ? 'location' : undefined}
              onClick={() => { setActive(id); requestAnimationFrame(() => {
                const section = formRef.current?.querySelector<HTMLElement>(`#${id}`);
                const heading = section?.querySelector<HTMLElement>('h2');
                heading?.focus({ preventScroll: true });
              }); }}>
              <Icon className="h-4 w-4" aria-hidden="true" />{title}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
}
