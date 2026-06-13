'use client';

import Link from 'next/link';
import { ArrowRight, FileJson2, GitBranch, ImagePlus, Sparkles, Workflow } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';

const workflowTools = [
  {
    name: 'Ideogram Workflow Builder',
    description: 'Build Ideogram 4 FP8 ComfyUI API workflows.',
    href: '/workflows/ideogram',
    status: 'Available',
    enabled: true,
    icon: Sparkles,
    tags: ['External ComfyUI', 'JSON export', 'Image import'],
  },
  {
    name: 'Flux Workflow Builder',
    description: 'Reusable external ComfyUI workflow tooling.',
    href: '#',
    status: 'Coming soon',
    enabled: false,
    icon: Workflow,
    tags: ['Coming Soon'],
  },
  {
    name: 'Image Edit Workflow',
    description: 'Reference-image and mask-driven builder flows.',
    href: '#',
    status: 'Coming soon',
    enabled: false,
    icon: ImagePlus,
    tags: ['Coming Soon'],
  },
];

export default function WorkflowsPage() {
  return (
    <>
      <TopBar className="h-14 border-gray-900 bg-[#02060a] px-4">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <GitBranch className="h-5 w-5 text-cyan-200" />
          <h1 className="truncate text-lg font-semibold text-gray-100">Workflows</h1>
        </div>
      </TopBar>

      <MainContent className="operator-scrollbar-none bg-[#02060a] px-3 pt-16 text-gray-100 sm:px-4">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4">
          <section className="operator-panel p-4">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-gray-100">Workflow Tools</h2>
                <div className="mt-1 text-sm text-gray-500">External ComfyUI builders and export utilities.</div>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {workflowTools.map(tool => {
                const Icon = tool.icon;
                const content = (
                  <div className="flex h-full flex-col rounded-sm border border-gray-800 bg-[#060a0f] p-4 transition-colors group-hover:border-cyan-800/70 group-hover:bg-[#071119]">
                    <div className="mb-4 flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-sm border border-gray-800 bg-gray-950">
                          <Icon className={tool.enabled ? 'h-5 w-5 text-cyan-200' : 'h-5 w-5 text-gray-600'} />
                        </div>
                        <div className="min-w-0">
                          <h3 className={tool.enabled ? 'truncate text-sm font-semibold text-gray-100' : 'truncate text-sm font-semibold text-gray-500'}>
                            {tool.name}
                          </h3>
                          <div className={tool.enabled ? 'mt-1 text-xs text-emerald-300' : 'mt-1 text-xs text-gray-600'}>
                            {tool.status}
                          </div>
                        </div>
                      </div>
                      {tool.enabled ? <ArrowRight className="h-4 w-4 text-gray-500 transition-colors group-hover:text-cyan-200" /> : null}
                    </div>

                    <p className={tool.enabled ? 'min-h-10 text-sm leading-5 text-gray-400' : 'min-h-10 text-sm leading-5 text-gray-600'}>
                      {tool.description}
                    </p>

                    <div className="mt-4 flex flex-wrap gap-2">
                      {tool.tags.map(tag => (
                        <span
                          key={tag}
                          className={tool.enabled ? 'rounded-sm border border-gray-800 bg-gray-950 px-2 py-1 text-xs text-gray-300' : 'rounded-sm border border-gray-900 bg-gray-950/50 px-2 py-1 text-xs text-gray-700'}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    {tool.enabled ? (
                      <span className="mt-5 inline-flex h-9 w-fit items-center gap-2 rounded-sm border border-cyan-800 bg-cyan-500 px-3 text-sm font-semibold text-gray-950 transition-colors group-hover:bg-cyan-400">
                        Open
                        <ArrowRight className="h-4 w-4" />
                      </span>
                    ) : null}
                  </div>
                );

                return tool.enabled ? (
                  <Link key={tool.name} href={tool.href} className="group block">
                    {content}
                  </Link>
                ) : (
                  <div key={tool.name} aria-disabled="true" className="opacity-80">
                    {content}
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </MainContent>
    </>
  );
}
