'use client';
import { useEffect, useRef, useState } from 'react';
import Link from '@/navigation/Link';
import { usePathname } from '@/navigation';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { ArrowUpRight, Search } from 'lucide-react';

const commands = [
  { title: 'Monitor', detail: 'Active runs, hardware and recent activity', href: '/dashboard' },
  { title: 'New training', detail: 'Training form and configuration editor', href: '/jobs/new' },
  { title: 'Continue LoRA', detail: 'Upload, refine and preserve an existing Krea 2 LoRA', href: '/loras/continue' },
  { title: 'Queue', detail: 'Training jobs, ordering and workers', href: '/jobs' },
  { title: 'Datasets', detail: 'Images, captions and dataset tools', href: '/datasets' },
  { title: 'Models', detail: 'Model library and checkpoints', href: '/models' },
  { title: 'Generate', detail: 'Image and video generation', href: '/generate' },
  { title: 'Settings', detail: 'Storage, workers and application preferences', href: '/settings' },
  { title: 'Tools', detail: 'Utilities and workflow tools', href: '/tools' },
  { title: 'Workflows', detail: 'Workflow library', href: '/workflows' },
  { title: 'Watermark', detail: 'Watermark tools', href: '/watermark' },
] as const;

export default function CommandPalette({
  open,
  onClose,
  onOpen,
}: {
  open: boolean;
  onClose: () => void;
  onOpen: () => void;
}) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const links = useRef<Array<HTMLAnchorElement | null>>([]);
  const searchInput = useRef<HTMLInputElement | null>(null);
  const pathname = usePathname();
  const previousPath = useRef(pathname);
  const filtered = commands.filter(item => `${item.title} ${item.detail}`.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'k' || event.altKey || event.isComposing)
        return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest('.monaco-editor') || (!open && target?.closest('[role="dialog"]'))) return;
      event.preventDefault();
      if (open) onClose();
      else onOpen();
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [open, onOpen, onClose]);
  useEffect(() => {
    if (previousPath.current !== pathname) onClose();
    previousPath.current = pathname;
  }, [pathname, onClose]);
  useEffect(() => {
    if (open) links.current[selected]?.scrollIntoView({ block: 'nearest' });
  }, [open, selected, query]);
  useEffect(() => {
    if (!open) return;
    // Close the search even when a document-level dirty-edit guard consumes the
    // ordinary link click. Navigation itself remains entirely owned by that link.
    const activate = (event: MouseEvent) => {
      if (event.target instanceof Element && event.target.closest('[data-command-link]')) onClose();
    };
    window.addEventListener('click', activate, true);
    return () => window.removeEventListener('click', activate, true);
  }, [open, onClose]);
  return (
    <Dialog open={open} onClose={onClose} initialFocus={searchInput} className="fixed inset-0 z-40">
      <DialogBackdrop className="fixed inset-0 bg-black/50" />
      <div className="fixed inset-0 flex items-start justify-center px-3 pt-[12vh]">
        <DialogPanel className="studio-command-panel w-full max-w-xl overflow-hidden rounded-md border border-gray-700 bg-gray-900 shadow-2xl">
          <DialogTitle className="sr-only">Search commands</DialogTitle>
          <div className="flex items-center gap-3 border-b border-gray-800 px-4 py-3">
            <Search size={18} className="text-gray-400" />
            <input
              ref={searchInput}
              aria-label="Search commands"
              aria-describedby="command-selected-result"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none"
              placeholder="Go to a workspace…"
              value={query}
              onChange={event => {
                setQuery(event.target.value);
                setSelected(0);
              }}
              onKeyDown={event => {
                if (event.nativeEvent.isComposing) return;
                if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                  event.preventDefault();
                  setSelected(index =>
                    Math.max(0, Math.min(filtered.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1))),
                  );
                }
                if (event.key === 'Enter') {
                  event.preventDefault();
                  links.current[selected]?.click();
                }
              }}
            />
            <button type="button" className="text-xs text-gray-400" onClick={onClose} aria-label="Close command search">
              Esc
            </button>
          </div>
          <nav aria-label="Command results" className="max-h-[55vh] overflow-auto p-1">
            {filtered.map((item, index) => (
              <Link
                data-command-link
                ref={node => {
                  links.current[index] = node;
                }}
                key={item.href}
                href={item.href}
                onClick={onClose}
                onFocus={() => setSelected(index)}
                onMouseEnter={() => setSelected(index)}
                aria-current={pathname === item.href ? 'page' : undefined}
                className={`flex items-center justify-between gap-4 rounded px-3 py-2.5 ${selected === index ? 'bg-brand-500/15 text-gray-100' : 'text-gray-300'}`}
              >
                <span>
                  <span className="block text-sm font-medium">{item.title}</span>
                  <span className="block text-xs text-gray-400">{item.detail}</span>
                </span>
                <ArrowUpRight size={15} />
              </Link>
            ))}
            {!filtered.length && (
              <p role="status" className="p-5 text-sm text-gray-400">
                No matching commands.
              </p>
            )}
          </nav>
          <p id="command-selected-result" role="status" className="sr-only">
            {filtered[selected]
              ? `Selected: ${filtered[selected].title}. Press Enter to open.`
              : 'No matching commands.'}
          </p>
          <div className="flex gap-4 border-t border-gray-800 px-4 py-2 text-xs text-gray-500">
            <span>↑ ↓ Select</span>
            <span>Enter Open</span>
            <span>Esc Close</span>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
