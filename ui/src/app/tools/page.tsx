import Link from 'next/link';
import { GitBranch, ShieldCheck, ArrowRight } from 'lucide-react';

export default function ToolsPage() {
  return (
    <div className="journey-page">
      <h1 className="journey-title">Tools</h1>
      <p className="mt-2 text-gray-400">Utilities for your training and generation workflow.</p>
      <div className="mt-8 grid gap-5 md:grid-cols-2">
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
          <Link key={item.href} href={item.href} className="journey-card p-6 hover:border-gray-500">
            <item.icon className="mb-5 h-6 w-6 text-[var(--studio-accent)]" aria-hidden="true" />
            <h2 className="text-xl font-semibold">{item.title}</h2>
            <p className="mt-2 text-gray-400">{item.description}</p>
            <span className="mt-5 inline-flex items-center gap-2">
              Open tool <ArrowRight size={16} />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
