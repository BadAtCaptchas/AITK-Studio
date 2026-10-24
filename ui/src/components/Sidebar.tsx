'use client';
import Link from '@/navigation/Link';
import { MoreHorizontal, LogOut, Settings, Plus, Search, Moon, Sun } from 'lucide-react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { usePathname } from '@/navigation';
import { useTheme } from './ThemeProvider';
import UpdaterStatus from './UpdaterStatus';
import { authRequiredState, logout } from '@/utils/api';

const navigation = [
  { name: 'Monitor', href: '/dashboard' },
  { name: 'Runs', href: '/jobs' },
  { name: 'Datasets', href: '/datasets' },
  { name: 'Models', href: '/models' },
  { name: 'Generate', href: '/generate' },
] as const;

export default function Sidebar({ onSearch }: { onSearch: () => void }) {
  const pathname = usePathname();
  const [authRequired] = authRequiredState.use();
  const { theme, toggleTheme } = useTheme();
  return (
    <header className="studio-app-header">
      <Link href="/dashboard" className="studio-wordmark" aria-label="AITK Studio home">
        <strong>AITK</strong> Studio
      </Link>
      <nav className="studio-app-navigation operator-scrollbar-none" aria-label="Main navigation">
        {navigation.map(item => (
          <Link
            key={item.href}
            href={item.href}
            aria-current={pathname === item.href || pathname.startsWith(`${item.href}/`) ? 'page' : undefined}
          >
            {item.name}
          </Link>
        ))}
      </nav>
      <div className="studio-app-actions">
        <button
          type="button"
          onClick={onSearch}
          className="operator-icon-button"
          aria-label="Search commands"
          title="Search commands (Ctrl / ⌘ K)"
        >
          <Search size={18} />
        </button>
        <Link href="/settings" className="operator-icon-button" aria-label="Settings" title="Settings">
          <Settings size={18} />
        </Link>
        <Menu>
          <MenuButton className="operator-icon-button" aria-label="More">
            <MoreHorizontal size={18} />
          </MenuButton>
          <MenuItems anchor="bottom end" className="studio-app-menu z-40">
            {[
              ['Tools', '/tools'],
              ['Workflows', '/workflows'],
              ['Watermark', '/watermark'],
            ].map(([label, href]) => (
              <MenuItem key={href}>
                <Link href={href} className="block px-3 py-2 text-sm data-[focus]:bg-gray-800">
                  {label}
                </Link>
              </MenuItem>
            ))}
            <div className="flex items-center gap-2 border-t border-gray-800 p-2">
              <MenuItem>
                <button
                  type="button"
                  onClick={toggleTheme}
                  className="operator-icon-button data-[focus]:bg-gray-800"
                  title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
                </button>
              </MenuItem>
              <UpdaterStatus compact minimal menuItem />
              {authRequired && (
                <MenuItem>
                  <button
                    type="button"
                    onClick={() => void logout()}
                    className="operator-icon-button data-[focus]:bg-gray-800"
                    aria-label="Log out"
                    title="Log out"
                  >
                    <LogOut size={16} />
                  </button>
                </MenuItem>
              )}
              <span className="ml-auto text-xs text-gray-500">v{import.meta.env.VITE_APP_VERSION}</span>
            </div>
          </MenuItems>
        </Menu>
        <Link href="/jobs/new" aria-label="New training" className="operator-button-primary studio-new-training">
          <Plus size={16} />
          <span>New training</span>
        </Link>
      </div>
    </header>
  );
}
