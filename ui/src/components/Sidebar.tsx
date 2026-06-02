'use client';

import Link from 'next/link';
import { Home, Settings, ListOrdered, Images, Plus, Wand2 } from 'lucide-react';
import { usePathname } from 'next/navigation';
import { FaXTwitter, FaDiscord, FaYoutube } from 'react-icons/fa6';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';
import ActiveJobWidget from './ActiveJobWidget';
import OstrisCloudBalance from './OstrisCloudBalance';
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

  const socialsBoxClass =
    'flex flex-col items-center justify-center rounded-sm p-1 hover:bg-gray-800 transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  const isActive = (href: string) => {
    if (href === '/jobs') {
      return pathname === '/jobs' || (pathname.startsWith('/jobs/') && !pathname.startsWith('/jobs/new'));
    }
    return pathname === href || pathname.startsWith(`${href}/`);
  };

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-gray-800 bg-gray-950 text-gray-100 md:h-screen md:w-60 md:border-b-0 md:border-r">
      <div className="flex items-center justify-between gap-3 px-3 py-2 md:block md:px-4 md:py-3">
        <h1 className="flex items-center text-sm leading-tight">
          <ThemeLogo />
          <span className="hidden flex-col uppercase sm:flex">
            <span className="font-bold">OstrisAI-Toolkit</span>
            <span className="text-gray-300">Revamped</span>
          </span>
        </h1>
        <div className="flex items-center gap-1 md:hidden">
          <UpdaterStatus compact />
          <ThemeToggle />
        </div>
      </div>
      <div className="hidden md:block">
        <OstrisCloudBalance />
      </div>
      <nav className="operator-scrollbar-none min-w-0 flex-1 overflow-x-auto md:overflow-y-auto md:overflow-x-hidden">
        <ul className="flex gap-1 px-2 pb-2 md:block md:space-y-1 md:py-3">
          {navigation.map(item => {
            const active = isActive(item.href);
            return (
              <li key={item.name} className="min-w-fit md:min-w-0">
                <Link
                  href={item.href}
                  aria-current={active ? 'page' : undefined}
                  className={`flex items-center gap-2 rounded-sm border px-3 py-2 text-sm transition-colors md:px-4 ${
                    active
                      ? 'border-cyan-800 bg-cyan-950/30 text-cyan-100'
                      : 'border-transparent text-gray-300 hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100'
                  }`}
                >
                  <item.icon className="h-4 w-4 flex-none" />
                  <span className="truncate">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="hidden md:block">
        <ActiveJobWidget />
      </div>
      <a
        href="https://ostris.com/support"
        target="_blank"
        rel="noreferrer"
        className="group hidden items-center space-x-2 px-4 py-3 text-gray-400 transition-colors hover:text-gray-200 md:flex"
      >
        <svg
          height="20"
          width="20"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          style={{ overflow: 'visible' }}
        >
          <path
            className="animate-heartbeat"
            d="m7 3c-1.5355 0-3.0784 0.5-4.25 1.7-2.3431 2.4-2.2788 6.1 0 8.5l9.25 9.8 9.25-9.8c2.279-2.4 2.343-6.1 0-8.5-2.343-2.3-6.157-2.3-8.5 0l-0.75 0.8-0.75-0.8c-1.172-1.2-2.7145-1.7-4.25-1.7z"
            fill="#c0392b"
          />
        </svg>
        <span className="uppercase text-sm font-medium tracking-wide">Support Ostris</span>
      </a>

      {/* Social links grid */}
      <div className="hidden border-t border-gray-800 px-1 py-1 md:block">
        <div className="grid grid-cols-4 gap-4">
          <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaDiscord className={socialIconClass} />
            {/* <span className="text-xs text-gray-500 mt-1">Discord</span> */}
          </a>
          <a href="https://www.youtube.com/@ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
            {/* <span className="text-xs text-gray-500 mt-1">YouTube</span> */}
          </a>
          <a href="https://x.com/ostrisai" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaXTwitter className={socialIconClass} />
            {/* <span className="text-xs text-gray-500 mt-1">X</span> */}
          </a>
          <ThemeToggle />
        </div>
      </div>
      <div className="hidden md:block">
        <UpdaterStatus />
      </div>
      <div className="hidden bg-gray-900 py-1 text-center text-[10px] text-gray-500 md:block">
        OstrisAI-Toolkit v{process.env.NEXT_PUBLIC_APP_VERSION}
      </div>
    </aside>
  );
};

export default Sidebar;
