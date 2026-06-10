'use client';

import Link from 'next/link';
import { Home, Images, ListOrdered, Plus, Settings, Wand2 } from 'lucide-react';
import { usePathname } from 'next/navigation';
import { FaDiscord } from 'react-icons/fa6';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';
import UpdaterStatus from './UpdaterStatus';

const Sidebar = () => {
  const pathname = usePathname();
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'New Job', href: '/jobs/new', icon: Plus },
    { name: 'Generate', href: '/generate', icon: Wand2 },
    { name: 'Queue', href: '/jobs', icon: ListOrdered },
    { name: 'Datasets', href: '/datasets', icon: Images },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const railButtonClass =
    'flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-white';

  const isActive = (href: string) => {
    if (href === '/jobs') {
      return pathname === '/jobs' || (pathname.startsWith('/jobs/') && !pathname.startsWith('/jobs/new'));
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-gray-900 bg-[#060a0f] text-gray-100 md:h-screen md:w-[124px] md:border-b-0 md:border-r">
      <div className="flex h-14 items-center justify-between gap-3 border-b border-gray-900 px-3 md:h-16 md:justify-center md:px-0">
        <Link href="/dashboard" className="flex min-w-0 items-center gap-2 md:flex-col md:gap-1" aria-label="Dashboard">
          <ThemeLogo className="mr-0 h-8" />
          <span className="truncate text-xs font-semibold text-gray-200 md:hidden">AI Toolkit</span>
        </Link>
        <div className="flex items-center gap-1 md:hidden">
          <UpdaterStatus compact minimal />
          <ThemeToggle variant="rail" />
        </div>
      </div>

      <nav className="operator-scrollbar-none min-w-0 flex-1 overflow-x-auto md:overflow-y-auto md:overflow-x-hidden">
        <ul className="flex gap-1 px-2 py-2 md:block md:space-y-2 md:px-2 md:py-5">
          {navigation.map(item => {
            const active = isActive(item.href);
            return (
              <li key={item.name} className="min-w-fit md:min-w-0">
                <Link
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  title={item.name}
                  className={`flex h-12 min-w-[88px] items-center justify-center gap-2 rounded-md border px-3 text-sm transition-colors md:h-[62px] md:min-w-0 md:flex-col md:gap-1 md:px-1 ${
                    active
                      ? 'border-blue-500/60 bg-blue-600/15 text-blue-100 shadow-[inset_3px_0_0_rgba(59,130,246,0.85)]'
                      : 'border-transparent text-gray-300 hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100'
                  }`}
                >
                  <item.icon className="h-5 w-5 flex-none" />
                  <span className="truncate text-xs">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="hidden border-t border-gray-900 px-2 py-2 md:block">
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
