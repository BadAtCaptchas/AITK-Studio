'use client';

import Link from 'next/link';
import { FaChevronLeft } from 'react-icons/fa';
import { TopBar, MainContent } from '@/components/layout';
import SecureRemoteCaptionPanel from '@/components/SecureRemoteCaptionPanel';

export default function SecureRemoteCaptioningPage() {
  return (
    <>
      <TopBar>
        <div>
          <Link href="/jobs" className="block px-3 text-gray-500 dark:text-gray-300">
            <FaChevronLeft />
          </Link>
        </div>
        <div>
          <h1 className="text-lg">Secure Remote Captioning</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <div className="mx-auto max-w-5xl">
          <SecureRemoteCaptionPanel />
        </div>
      </MainContent>
    </>
  );
}
