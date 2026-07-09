import Link from 'next/link';
import { ArrowLeft, FileQuestion } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="flex min-h-full items-center justify-center bg-gray-950 p-6 text-gray-100">
      <div className="w-full max-w-lg rounded-md border border-gray-800 bg-gray-900/50 p-6">
        <FileQuestion className="h-7 w-7 text-cyan-300" aria-hidden="true" />
        <h1 className="mt-4 text-lg font-semibold">Page not found</h1>
        <p className="mt-2 text-sm leading-6 text-gray-400">
          The requested workspace item may have moved, been archived, or been removed.
        </p>
        <Link href="/dashboard" className="operator-button mt-5 h-9">
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back to dashboard
        </Link>
      </div>
    </div>
  );
}
