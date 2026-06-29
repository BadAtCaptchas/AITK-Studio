'use client';

import Link from 'next/link';
import { FolderKanban, GitBranch, Home, Images, ListOrdered, Plus, Settings, ShieldCheck, Wand2 } from 'lucide-react';
import { usePathname } from 'next/navigation';
import { FaDiscord } from 'react-icons/fa6';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';
import UpdaterStatus from './UpdaterStatus';
import useSettings from '@/hooks/useSettings';

const Sidebar = () => {
  const pathname = usePathname();
  const { settings } = useSettings();
  const projectsEnabled = settings.PROJECTS_ENABLED !== 'false';
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Projects', href: '/projects', icon: FolderKanban },
    { name: 'New Job', href: '/jobs/new', icon: Plus },
    { name: 'Generate', href: '/generate', icon: Wand2 },
    { name: 'Watermark', href: '/watermark', icon: ShieldCheck },
    { name: 'Workflows', href: '/workflows', icon: GitBranch },
    { name: 'Queue', href: '/jobs', icon: ListOrdered },
    { name: 'Datasets', href: '/datasets', icon: Images },
    { name: 'Settings', href: '/settings', icon: Settings },
  ].filter(item => item.name !== 'Projects' || projectsEnabled);

  const railButtonClass =
    'flex h-8 w-8 items-center justify-center rounded-sm border border-transparent text-gray-500 transition-colors hover:border-gray-800 hover:bg-gray-900/60 hover:text-gray-100';

  const isActive = (href: string) => {
    if (href === '/jobs') {
      return pathname === '/jobs' || (pathname.startsWith('/jobs/') && !pathname.startsWith('/jobs/new'));
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-gray-900 bg-gray-950 text-gray-100 md:h-screen md:w-[124px] md:border-b-0 md:border-r">
      <div className="flex h-14 items-center justify-between gap-3 border-b border-gray-900/80 px-3 md:h-16 md:justify-center md:px-0">
        <Link href="/dashboard" className="flex min-w-0 items-center gap-2 md:flex-col md:gap-1" aria-label="Dashboard">
          <ThemeLogo className="mr-0 h-8" />
          <span className="truncate text-xs font-semibold text-gray-200 md:hidden">AITK Studio</span>
        </Link>
        <div className="flex items-center gap-1 md:hidden">
          <UpdaterStatus compact minimal />
          <ThemeToggle variant="rail" />
        </div>
      </div>

      <nav className="operator-scrollbar-none min-w-0 flex-1 overflow-x-auto md:overflow-y-auto md:overflow-x-hidden">
        <ul className="flex gap-1 px-2 py-2 md:block md:space-y-1.5 md:px-2 md:py-5">
          {navigation.map(item => {
            const active = isActive(item.href);
            return (
              <li key={item.name} className="min-w-fit md:min-w-0">
                <Link
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  title={item.name}
                  className={`group flex h-12 min-w-[88px] items-center justify-center gap-2 rounded-sm border px-3 text-sm transition-colors md:h-[62px] md:min-w-0 md:flex-col md:gap-1 md:px-1 ${
                    active
                      ? 'border-gray-800 bg-gray-900/45 text-gray-100 shadow-[inset_0_-2px_0_rgba(34,211,238,0.85)] md:shadow-[inset_2px_0_0_rgba(34,211,238,0.85)]'
                      : 'border-transparent text-gray-400 hover:border-gray-800 hover:bg-gray-900/45 hover:text-gray-200'
                  }`}
                >
                  <item.icon className={`h-5 w-5 flex-none ${active ? 'text-cyan-200' : 'text-gray-500 group-hover:text-gray-300'}`} />
                  <span className="truncate text-xs font-medium">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="hidden border-t border-gray-900/80 px-2 py-2 md:block">
        <div className="flex items-center justify-center gap-1.5">
          <a
            href="https://discord.gg/umF6SfKRtm"
            target="_blank"
            rel="noreferrer"
            className={railButtonClass}
            title="Discord"
            aria-label="Discord"
          >
            <FaDiscord className="h-4 w-4" />
          </a>
          <UpdaterStatus compact minimal />
          <ThemeToggle variant="rail" />
        </div>
        <div className="mt-1 text-center text-[10px] text-gray-500">v{process.env.NEXT_PUBLIC_APP_VERSION}</div>
      </div>
    </aside>
  );
};

export default Sidebar;
