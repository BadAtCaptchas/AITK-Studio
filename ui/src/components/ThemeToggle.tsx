'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from './ThemeProvider';

type ThemeToggleProps = {
  variant?: 'default' | 'rail';
};

const ThemeToggle = ({ variant = 'default' }: ThemeToggleProps) => {
  const { theme, toggleTheme } = useTheme();

  // Button styled as the opposite theme so it stands out
  const buttonClass =
    theme === 'dark'
      ? 'bg-[rgb(215,215,215)] hover:bg-[rgb(195,195,195)] text-[rgb(82,82,82)]'
      : 'bg-[rgb(23,23,23)] hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)]';
  const railButtonClass =
    'h-8 w-8 rounded-md border border-transparent text-gray-400 hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100';
  const iconClass = variant === 'rail' ? 'h-4 w-4' : 'h-5 w-5';

  return (
    <button
      onClick={toggleTheme}
      className={`flex items-center justify-center transition-colors ${variant === 'rail' ? railButtonClass : `rounded-lg p-1 ${buttonClass}`}`}
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {theme === 'dark' ? <Sun className={iconClass} /> : <Moon className={iconClass} />}
    </button>
  );
};

export default ThemeToggle;
