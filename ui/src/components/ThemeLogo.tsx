'use client';

import classNames from 'classnames';
import { useTheme } from './ThemeProvider';

const ThemeLogo = ({ className }: { className?: string }) => {
  const { theme } = useTheme();
  const src = theme === 'dark' ? '/aitk-studio-logo-light.png' : '/aitk-studio-logo-dark.png';

  return <img src={src} alt="AITK Studio" className={classNames('inline h-7 w-auto', className)} />;
};

export default ThemeLogo;
