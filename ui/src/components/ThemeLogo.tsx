'use client';

import classNames from 'classnames';
import { useTheme } from './ThemeProvider';

const ThemeLogo = ({ className }: { className?: string }) => {
  const { theme } = useTheme();
  const src = theme === 'dark' ? '/ostris_logo.png' : '/ostris_logo_black.png';

  return <img src={src} alt="OstrisAI-Toolkit Revamped" className={classNames('inline h-7 w-auto', className)} />;
};

export default ThemeLogo;
