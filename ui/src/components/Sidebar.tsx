'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Bell,
  Boxes,
  Database,
  Gauge,
  Heart,
  Home,
  ListTree,
  Plus,
  Settings,
  ShieldCheck,
} from 'lucide-react';
import { FaDiscord, FaXTwitter, FaYoutube } from 'react-icons/fa6';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'New Job', href: '/jobs/new', icon: Plus },
  { name: 'Queue', href: '/jobs', icon: ListTree },
  { name: 'Datasets', href: '/datasets', icon: Database },
  { name: 'Jobs', href: '/jobs', icon: Gauge },
  { name: 'Models', href: '/models', icon: Boxes },
  { name: 'Evaluations', href: '/evaluations', icon: ShieldCheck },
  { name: 'Alerts', href: '/alerts', icon: Bell },
  { name: 'Settings', href: '/settings', icon: Settings },
];

function isActive(pathname: string, href: string, name: string) {
  if (name === 'New Job') return pathname.startsWith('/jobs/new');
  if (name === 'Jobs') return pathname.startsWith('/jobs/') && !pathname.startsWith('/jobs/new');
  if (name === 'Queue') return pathname === '/jobs';
  return pathname === href || pathname.startsWith(`${href}/`);
}

const socialClass = 'flex h-7 w-7 items-center justify-center rounded border border-white/5 text-gray-500 hover:bg-white/5 hover:text-gray-200';

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-14 shrink-0 flex-col border-r border-white/10 bg-black text-gray-200 sm:w-56">
      <div className="flex h-14 items-center gap-3 border-b border-white/10 px-3">
        <ThemeLogo />
        <div className="hidden min-w-0 sm:block">
          <div className="truncate text-sm font-semibold tracking-wide text-white">OSTRIS</div>
          <div className="truncate text-[11px] uppercase text-gray-500">AI-Toolkit</div>
        </div>
      </div>

      <nav className="flex-1 py-3">
        <ul className="space-y-1 px-2">
          {navigation.map(item => {
            const active = isActive(pathname, item.href, item.name);
            return (
              <li key={`${item.name}-${item.href}`}>
                <Link
                  href={item.href}
                  className={[
                    'group flex h-10 items-center gap-3 rounded px-3 text-sm transition-colors',
                    active
                      ? 'bg-blue-500/10 text-blue-400 ring-1 ring-blue-500/20'
                      : 'text-gray-400 hover:bg-white/5 hover:text-gray-100',
                  ].join(' ')}
                  title={item.name}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  <span className="hidden truncate sm:inline">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="space-y-3 border-t border-white/10 p-3">
        <a
          href="https://ostris.com/support"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 text-[11px] uppercase text-gray-500 hover:text-gray-200"
          title="Support AI-Toolkit"
        >
          <Heart className="h-4 w-4 fill-red-500 text-red-500" />
          <span className="hidden sm:inline">Support AI-Toolkit</span>
        </a>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-4">
          <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialClass} title="Discord">
            <FaDiscord className="h-4 w-4" />
          </a>
          <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialClass} title="YouTube">
            <FaYoutube className="h-4 w-4" />
          </a>
          <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialClass} title="X">
            <FaXTwitter className="h-4 w-4" />
          </a>
          <div className={socialClass}>
            <ThemeToggle />
          </div>
        </div>
        <div className="hidden text-[11px] text-gray-600 sm:block">v2.6.0</div>
      </div>
    </aside>
  );
}
