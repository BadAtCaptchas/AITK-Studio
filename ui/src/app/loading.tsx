import { Loader2 } from 'lucide-react';

export default function Loading() {
  return (
    <div className="flex min-h-full items-center justify-center bg-gray-950 p-6 text-gray-300" role="status">
      <Loader2 className="mr-2 h-5 w-5 animate-spin" aria-hidden="true" />
      Loading workspace
      <span className="sr-only">Please wait</span>
    </div>
  );
}
