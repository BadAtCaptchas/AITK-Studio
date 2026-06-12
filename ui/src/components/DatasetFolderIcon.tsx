'use client';

import { useEffect, useState } from 'react';
import classNames from 'classnames';
import { CheckCircle2, Cloud, Image, LockKeyhole } from 'lucide-react';

type DatasetFolderIconSize = 'sm' | 'md' | 'lg';

interface DatasetFolderIconProps {
  encrypted?: boolean;
  unlocked?: boolean;
  remote?: boolean;
  previewSrc?: string | null;
  size?: DatasetFolderIconSize;
  className?: string;
}

const sizeClasses: Record<
  DatasetFolderIconSize,
  {
    wrap: string;
    tab: string;
    body: string;
    image: string;
    preview: string;
    glyph: string;
    badge: string;
    badgeIcon: string;
  }
> = {
  sm: {
    wrap: 'h-9 w-11',
    tab: 'left-1 top-1 h-2.5 w-5',
    body: 'inset-x-0 bottom-0 h-7',
    image: 'right-1.5 top-3 h-4 w-4',
    preview: 'left-2 right-2 top-3 h-3.5',
    glyph: 'h-2.5 w-2.5',
    badge: '-right-1 -bottom-1 h-4 w-4',
    badgeIcon: 'h-2.5 w-2.5',
  },
  md: {
    wrap: 'h-12 w-14',
    tab: 'left-1.5 top-1 h-3.5 w-6',
    body: 'inset-x-0 bottom-0 h-9',
    image: 'right-2 top-4 h-5 w-5',
    preview: 'left-2.5 right-2.5 top-3 h-5',
    glyph: 'h-3 w-3',
    badge: '-right-1 -bottom-1 h-5 w-5',
    badgeIcon: 'h-3 w-3',
  },
  lg: {
    wrap: 'h-20 w-24',
    tab: 'left-2 top-2 h-5 w-10',
    body: 'inset-x-0 bottom-0 h-14',
    image: 'right-3 top-7 h-8 w-8',
    preview: 'left-4 right-4 top-5 h-8',
    glyph: 'h-5 w-5',
    badge: '-right-1.5 -bottom-1.5 h-7 w-7',
    badgeIcon: 'h-4 w-4',
  },
};

export default function DatasetFolderIcon({
  encrypted = false,
  unlocked = false,
  remote = false,
  previewSrc = null,
  size = 'md',
  className,
}: DatasetFolderIconProps) {
  const classes = sizeClasses[size];
  const [previewFailed, setPreviewFailed] = useState(false);
  const showPreview = !!previewSrc && !previewFailed;
  const badgeClass = encrypted
    ? unlocked
      ? 'border-emerald-500/60 bg-emerald-950 text-emerald-200'
      : 'border-cyan-500/50 bg-cyan-950 text-cyan-200'
      : remote
        ? 'border-sky-500/50 bg-sky-950 text-sky-200'
        : 'border-amber-300/40 bg-amber-950 text-amber-100';

  useEffect(() => {
    setPreviewFailed(false);
  }, [previewSrc]);

  return (
    <span className={classNames('relative block flex-none', classes.wrap, className)} aria-hidden="true">
      <span
        className={classNames(
          'absolute rounded-t-sm border border-amber-300/50 bg-amber-300 shadow-sm shadow-black/20',
          classes.tab,
        )}
      />
      <span
        className={classNames(
          'absolute rounded-sm border border-amber-300/55 bg-gradient-to-br from-amber-300 via-yellow-400 to-orange-500 shadow-md shadow-black/30',
          classes.body,
        )}
      >
        <span className="absolute inset-x-1 top-1 h-px bg-white/45" />
        {showPreview ? (
          <span
            className={classNames(
              'absolute overflow-hidden rounded-sm border border-amber-100/70 bg-amber-50 shadow-sm shadow-orange-950/30',
              classes.preview,
            )}
          >
            <img
              src={previewSrc}
              alt=""
              className="h-full w-full object-cover"
              loading="lazy"
              decoding="async"
              onError={() => setPreviewFailed(true)}
            />
          </span>
        ) : (
          <>
            <span className="absolute left-1.5 top-1.5 h-3 w-3 rounded-full bg-white/25" />
            <span className="absolute bottom-1 left-1.5 h-3 w-5 rounded-sm bg-orange-700/20" />
          </>
        )}
        <span className="absolute inset-x-1 bottom-1 h-2 rounded-sm bg-orange-700/15" />
      </span>
      {!showPreview && (
        <span
          className={classNames(
            'absolute flex items-center justify-center rounded-sm border border-sky-200/70 bg-gradient-to-br from-sky-200 via-cyan-300 to-blue-500 text-blue-950 shadow-sm shadow-black/30',
            classes.image,
          )}
        >
          <Image className={classes.glyph} strokeWidth={2.4} />
        </span>
      )}
      {(encrypted || remote) && (
        <span
          className={classNames(
            'absolute flex items-center justify-center rounded-full border shadow-sm shadow-black/40',
            classes.badge,
            badgeClass,
          )}
        >
          {encrypted ? (
            unlocked ? (
              <CheckCircle2 className={classes.badgeIcon} strokeWidth={2.3} />
            ) : (
              <LockKeyhole className={classes.badgeIcon} strokeWidth={2.3} />
            )
          ) : (
            <Cloud className={classes.badgeIcon} strokeWidth={2.3} />
          )}
        </span>
      )}
    </span>
  );
}
