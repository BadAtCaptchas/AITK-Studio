import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';
import os from 'os';
import { CaptionDatasetModal } from '@/components/CaptionDatasetModal';
import MergeLoRAsModal from '@/components/MergeLoRAsModal';
import AppShell from '@/components/AppShell';

export const dynamic = 'force-dynamic';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AITK Studio',
  description: 'A training studio for diffusion models, datasets, and generation workflows.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Check if the AI_TOOLKIT_AUTH environment variable is set
  const authRequired = process.env.AI_TOOLKIT_AUTH ? true : false;

  const platform = os.platform();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="apple-mobile-web-app-title" content="AITK Studio" />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var theme = localStorage.getItem('theme') || 'dark';
                if (theme === 'dark') document.documentElement.classList.add('dark');
              })();
            `,
          }}
        />
      </head>
      <body className={inter.className}>
        <script dangerouslySetInnerHTML={{ __html: `window.server_platform = "${platform}";` }} />
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <AppShell>{children}</AppShell>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
        <CaptionDatasetModal />
        <MergeLoRAsModal />
      </body>
    </html>
  );
}
