import Link from 'next/link';
import { GitBranch, ShieldCheck, ArrowRight, Wrench } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';

export default function ToolsPage() {
  return (
    <>
      <TopBar>
        <Wrench size={16} className="text-gray-400" aria-hidden="true" />
        <h1 className="text-base font-semibold">Tools</h1>
      </TopBar>
      <MainContent>
        <div className="divide-y divide-gray-800 border-y border-gray-800">
          {[
            {
              href: '/workflows',
              title: 'Workflows',
              description: 'Build and run reusable generation workflows.',
              icon: GitBranch,
            },
            {
              href: '/watermark',
              title: 'Watermark',
              description: 'Manage and verify model watermarks.',
              icon: ShieldCheck,
            },
          ].map(item => (
            <Link
              key={item.href}
              href={item.href}
              className="group flex items-center gap-4 px-4 py-4 hover:bg-gray-900/60"
            >
              <item.icon className="h-5 w-5 shrink-0 text-gray-400" aria-hidden="true" />
              <div className="min-w-0 flex-1">
                <h2 className="text-sm font-medium">{item.title}</h2>
                <p className="mt-1 text-sm text-gray-400">{item.description}</p>
              </div>
              <ArrowRight size={16} className="shrink-0 text-gray-500 group-hover:text-brand-300" aria-hidden="true" />
            </Link>
          ))}
        </div>
      </MainContent>
    </>
  );
}
