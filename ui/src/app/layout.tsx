import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';
import os from 'os';
import { CaptionDatasetModal } from '@/components/CaptionDatasetModal';

export const dynamic = 'force-dynamic';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'OstrisAI-Toolkit Revamped',
  description: 'A revamped fork of Ostris AI Toolkit for training diffusion models.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Check if the AI_TOOLKIT_AUTH environment variable is set
  const authRequired = process.env.AI_TOOLKIT_AUTH ? true : false;

  const platform = os.platform();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="apple-mobile-web-app-title" content="AITK Revamped" />
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
            <div className="flex h-screen bg-black">
              <Sidebar />
              <main className="flex-1 overflow-auto bg-black text-gray-100 relative">
                <Suspense>{children}</Suspense>
              </main>
            </div>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
        <CaptionDatasetModal />
      </body>
    </html>
  );
}
