'use client';

import Link from 'next/link';
import { ArrowRight, GitBranch, Sparkles } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';

export default function WorkflowsPage() {
  return (
    <>
      <TopBar>
        <GitBranch size={16} className="text-gray-400" aria-hidden="true" />
        <h1 className="text-base font-semibold">Workflows</h1>
        <div className="flex-1" />
        <Link href="/tools" className="operator-button">
          All tools
        </Link>
      </TopBar>
      <MainContent>
        <div className="mb-3 text-xs text-gray-400">External ComfyUI builders and JSON export</div>
        <div className="border-y border-gray-800">
          <Link href="/workflows/ideogram" className="group flex items-center gap-4 px-4 py-4 hover:bg-gray-900/60">
            <Sparkles size={20} className="shrink-0 text-gray-400" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <h2 className="text-sm font-medium">Ideogram Workflow Builder</h2>
              <p className="mt-1 text-sm text-gray-400">
                Ideogram 4 FP8 ComfyUI API workflows, image import, and JSON export.
              </p>
            </div>
            <ArrowRight size={16} className="shrink-0 text-gray-500 group-hover:text-brand-300" aria-hidden="true" />
          </Link>
        </div>
      </MainContent>
    </>
  );
}
