'use client';

import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="flex min-h-full items-center justify-center bg-gray-950 p-6 text-gray-100">
      <div className="w-full max-w-lg rounded-md border border-rose-900/70 bg-rose-950/20 p-6">
        <AlertTriangle className="h-7 w-7 text-rose-300" aria-hidden="true" />
        <h1 className="mt-4 text-lg font-semibold">This page could not be loaded</h1>
        <p className="mt-2 text-sm leading-6 text-gray-400">
          The current operation stopped safely. Retry the page, then check the job or application log if the problem
          continues.
        </p>
        <button type="button" onClick={reset} className="operator-button mt-5 h-9">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          Try again
        </button>
      </div>
    </div>
  );
}
