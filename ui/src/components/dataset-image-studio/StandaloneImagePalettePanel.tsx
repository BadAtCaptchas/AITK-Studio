'use client';

import classNames from 'classnames';
import { Check, CircleAlert, Info, LoaderCircle, RefreshCw, TriangleAlert } from 'lucide-react';
import { useMemo } from 'react';
import { normalizeHexColor } from './utils';

export type PaletteSwatch = {
  color: string;
  /** Fraction of image pixels represented by this color, from 0 to 1. */
  coverage?: number | null;
  /** Extraction confidence for this color, from 0 to 1. */
  confidence?: number | null;
};

export type PaletteLayer = {
  elementIndex: number;
  label: string;
  colorPalette: readonly string[];
};

export type PaletteNearDuplicateIssue = {
  id: string;
  kind: 'near-duplicate';
  colors: readonly [string, string];
  tone?: 'warning' | 'info';
};

export type PaletteLowCoverageIssue = {
  id: string;
  kind: 'low-coverage';
  color: string;
  /** Fraction of image pixels represented by this color, from 0 to 1. */
  coverage: number;
};

export type PaletteUnassignedIssue = {
  id: string;
  kind: 'unassigned';
  color: string;
};

export type PaletteReviewIssue = PaletteNearDuplicateIssue | PaletteLowCoverageIssue | PaletteUnassignedIssue;

export type StandaloneImagePalettePanelProps = {
  palette: readonly PaletteSwatch[];
  layers: readonly PaletteLayer[];
  reviewIssues: readonly PaletteReviewIssue[];
  /** Overall extraction confidence, from 0 to 1. */
  confidence: number | null;
  /** Portion of the image represented by the palette, from 0 to 1. */
  coverage: number | null;
  selectedColor: string | null;
  isExtracting?: boolean;
  disabled?: boolean;
  disabledReason?: string;
  extractionMessage?: string;
  onExtractPalette: () => void;
  onSelectColor: (color: string) => void;
  onMergeColors: (issue: PaletteNearDuplicateIssue) => void;
  onReviewColor: (issue: PaletteLowCoverageIssue) => void;
  onAssignColor: (issue: PaletteUnassignedIssue) => void;
  onToggleLayerColor: (elementIndex: number, color: string, assigned: boolean) => void;
};

type NormalizedSwatch = PaletteSwatch & { color: string };

function clampRatio(value: number | null | undefined) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return Math.min(1, Math.max(0, value));
}

function formatConfidence(value: number | null) {
  const ratio = clampRatio(value);
  if (ratio == null) return 'Not analyzed';
  const label = ratio >= 0.85 ? 'High' : ratio >= 0.65 ? 'Medium' : 'Low';
  return `${label} (${ratio.toFixed(2)})`;
}

function formatCoverage(value: number | null) {
  const ratio = clampRatio(value);
  return ratio == null ? 'Not analyzed' : `${Math.round(ratio * 100)}%`;
}

function formatIssueCoverage(value: number) {
  const ratio = clampRatio(value) ?? 0;
  const precision = ratio > 0 && ratio < 0.1 ? 1 : 0;
  return `${(ratio * 100).toFixed(precision)}%`;
}

function issueColors(issue: PaletteReviewIssue) {
  const rawColors = issue.kind === 'near-duplicate' ? issue.colors : [issue.color];
  return rawColors.flatMap(color => {
    const normalized = normalizeHexColor(color);
    return normalized ? [normalized] : [];
  });
}

function issueDescription(issue: PaletteReviewIssue) {
  const colors = issueColors(issue);
  if (issue.kind === 'near-duplicate') {
    return colors.length === 2
      ? `${colors[0]} and ${colors[1]} are very similar`
      : 'Two palette colors are very similar';
  }
  if (issue.kind === 'low-coverage') {
    return `${colors[0] || 'This color'} covers ${formatIssueCoverage(issue.coverage)} of the image`;
  }
  return `${colors[0] || 'This color'} is not assigned to any layer`;
}

function issueTitle(issue: PaletteReviewIssue) {
  if (issue.kind === 'near-duplicate') return 'Near duplicate';
  if (issue.kind === 'low-coverage') return 'Low coverage';
  return 'Unassigned color';
}

function SwatchPreview({ color, compact = false }: { color: string; compact?: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={classNames(
        'block flex-none rounded-[2px] border border-white/10 shadow-[0_1px_2px_rgba(0,0,0,0.35)]',
        compact ? 'h-6 w-6' : 'h-11 w-full',
      )}
      style={{ backgroundColor: color }}
    />
  );
}

function ReviewIssueRow({
  issue,
  disabled,
  onMergeColors,
  onReviewColor,
  onAssignColor,
}: {
  issue: PaletteReviewIssue;
  disabled: boolean;
  onMergeColors: (issue: PaletteNearDuplicateIssue) => void;
  onReviewColor: (issue: PaletteLowCoverageIssue) => void;
  onAssignColor: (issue: PaletteUnassignedIssue) => void;
}) {
  const colors = issueColors(issue);
  const warning = issue.kind === 'low-coverage' || (issue.kind === 'near-duplicate' && issue.tone !== 'info');
  const actionLabel = issue.kind === 'near-duplicate' ? 'Merge' : issue.kind === 'low-coverage' ? 'Review' : 'Assign';

  return (
    <div className="grid min-h-[45px] grid-cols-[18px_minmax(0,1fr)_auto_80px] items-center gap-2 rounded-[4px] border border-[#27323c] bg-[#07131c] px-2.5 py-1">
      <span className={warning ? 'text-amber-400' : 'text-cyan-400'} aria-hidden="true">
        {issue.kind === 'low-coverage' ? (
          <TriangleAlert className="h-4 w-4" />
        ) : warning ? (
          <CircleAlert className="h-4 w-4" />
        ) : (
          <Info className="h-4 w-4" />
        )}
      </span>

      <div className="min-w-0">
        <div className="truncate text-xs font-medium leading-4 text-gray-100">{issueTitle(issue)}</div>
        <div className="truncate text-[10px] leading-3.5 text-gray-500" title={issueDescription(issue)}>
          {issueDescription(issue)}
        </div>
      </div>

      <div className="flex items-center gap-1" aria-label={`Affected colors: ${colors.join(', ')}`}>
        {colors.map(color => (
          <SwatchPreview key={color} color={color} compact />
        ))}
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          if (issue.kind === 'near-duplicate') onMergeColors(issue);
          else if (issue.kind === 'low-coverage') onReviewColor(issue);
          else onAssignColor(issue);
        }}
        aria-label={`${actionLabel} ${issueTitle(issue).toLocaleLowerCase()} issue`}
        className="flex h-7 items-center justify-center rounded-[4px] border border-[#34414d] bg-[#071019] px-3 text-xs font-medium text-gray-100 transition-colors hover:border-gray-500 hover:bg-[#0d1a24] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:cursor-not-allowed disabled:text-gray-600"
      >
        {actionLabel}
      </button>
    </div>
  );
}

export function StandaloneImagePalettePanel({
  palette,
  layers,
  reviewIssues,
  confidence,
  coverage,
  selectedColor,
  isExtracting = false,
  disabled = false,
  disabledReason = '',
  extractionMessage = '',
  onExtractPalette,
  onSelectColor,
  onMergeColors,
  onReviewColor,
  onAssignColor,
  onToggleLayerColor,
}: StandaloneImagePalettePanelProps) {
  const normalizedPalette = useMemo<NormalizedSwatch[]>(() => {
    const seen = new Set<string>();
    return palette.flatMap(swatch => {
      const color = normalizeHexColor(swatch.color);
      if (!color || seen.has(color)) return [];
      seen.add(color);
      return [{ ...swatch, color }];
    });
  }, [palette]);
  const normalizedSelectedColor = normalizeHexColor(selectedColor);
  const extractDisabled = disabled || isExtracting;
  const extractTitle = disabledReason || (isExtracting ? 'Extracting image palette' : 'Extract image palette');

  return (
    <section
      className="flex h-full min-h-0 flex-col overflow-hidden bg-[#06111a] text-gray-100"
      aria-labelledby="image-palette-heading"
    >
      <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto px-1 pb-1.5 pt-[10px]">
        <div className="flex h-[34px] items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-2">
            <h2 id="image-palette-heading" className="truncate text-sm font-semibold tracking-tight text-gray-100">
              Image palette
            </h2>
            <Info className="h-3.5 w-3.5 flex-none text-gray-500" aria-hidden="true" />
            {extractionMessage ? (
              <span className="min-w-0 truncate text-[10px] text-gray-500" role="status" aria-live="polite">
                {extractionMessage}
              </span>
            ) : null}
          </div>
          <button
            type="button"
            disabled={extractDisabled}
            onClick={onExtractPalette}
            title={extractTitle}
            className="mr-1 flex h-8 w-[151px] flex-none items-center justify-center gap-2 rounded-[4px] border border-cyan-300/60 bg-cyan-400 text-xs font-semibold text-[#03121a] shadow-[0_1px_8px_rgba(34,211,238,0.12)] transition-colors hover:bg-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-100 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-800 disabled:text-gray-500"
          >
            {isExtracting ? (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            Extract palette
          </button>
        </div>

        {normalizedPalette.length > 0 ? (
          <div className="operator-scrollbar-none mt-[3px] overflow-x-auto pb-1">
            <div
              className="grid gap-[5px]"
              style={{
                width: normalizedPalette.length <= 8 ? '528px' : `${normalizedPalette.length * 62 + (normalizedPalette.length - 1) * 5}px`,
                gridTemplateColumns: `repeat(${normalizedPalette.length}, minmax(54px, 1fr))`,
              }}
            >
              {normalizedPalette.map(({ color }) => {
                const selected = normalizedSelectedColor === color;
                return (
                  <button
                    key={color}
                    type="button"
                    onClick={() => onSelectColor(color)}
                    aria-label={`Select palette color ${color}`}
                    aria-pressed={selected}
                    className="group min-w-0 rounded-[3px] text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
                  >
                    <span
                      className={classNames(
                        'block rounded-[3px] border p-[1px] transition-colors',
                        selected ? 'border-cyan-300' : 'border-transparent group-hover:border-gray-500',
                      )}
                    >
                      <SwatchPreview color={color} />
                    </span>
                    <span
                      className={classNames(
                        'mt-1 block truncate text-center font-mono text-[9px]',
                        selected ? 'text-cyan-200' : 'text-gray-400',
                      )}
                    >
                      {color}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex h-[63px] items-center justify-center rounded-[4px] border border-dashed border-[#33414d] text-xs text-gray-500">
            Extract a palette to review image colors
          </div>
        )}

        <dl className="mt-[6px] grid h-[34px] grid-cols-2 items-center border-y border-[#26313b] text-[11px]">
          <div className="grid grid-cols-[auto_1fr] items-center gap-6 pr-5">
            <dt className="text-gray-400">Confidence</dt>
            <dd className="truncate font-medium text-gray-100">{formatConfidence(confidence)}</dd>
          </div>
          <div className="grid grid-cols-[auto_1fr] items-center gap-6 pl-5">
            <dt className="text-gray-400">Coverage</dt>
            <dd className="truncate font-medium text-gray-100">{formatCoverage(coverage)}</dd>
          </div>
        </dl>

        <div className="flex h-[32px] items-center gap-2">
          <h3 className="text-xs font-medium text-gray-200">Review issues</h3>
          <span
            className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-amber-950/70 px-1.5 text-[10px] font-semibold tabular-nums text-amber-300"
            aria-label={`${reviewIssues.length} review issues`}
          >
            {reviewIssues.length}
          </span>
        </div>

        {reviewIssues.length > 0 ? (
          <div className="-ml-1 w-full space-y-[3px]">
            {reviewIssues.map(issue => (
              <ReviewIssueRow
                key={issue.id}
                issue={issue}
                disabled={disabled}
                onMergeColors={onMergeColors}
                onReviewColor={onReviewColor}
                onAssignColor={onAssignColor}
              />
            ))}
          </div>
        ) : (
          <div className="flex h-[45px] items-center justify-center rounded-[4px] border border-[#27323c] bg-[#07131c] text-xs text-gray-500">
            No palette issues found
          </div>
        )}

        <div className="flex h-[30px] items-center gap-2">
          <h3 className="text-xs font-medium text-gray-200">Assign to layers</h3>
          <Info className="h-3.5 w-3.5 text-gray-500" aria-hidden="true" />
        </div>

        {normalizedPalette.length > 0 && layers.length > 0 ? (
          <div className="operator-scrollbar-none -ml-1 w-full overflow-x-auto rounded-[4px] border border-[#2b3741]">
            <table className="w-full border-separate border-spacing-0 text-[10px]">
              <thead>
                <tr className="h-[23px] bg-[#071019]">
                  <th
                    scope="col"
                    className="sticky left-0 z-10 min-w-[138px] border-b border-r border-[#2b3741] bg-[#071019] px-2.5 text-left font-medium text-gray-300"
                  >
                    Layer
                  </th>
                  {normalizedPalette.map(({ color }) => (
                    <th
                      key={color}
                      scope="col"
                      className={classNames(
                        'min-w-[50px] border-b border-r border-[#2b3741] px-1 text-center font-mono text-[9px] last:border-r-0',
                        normalizedSelectedColor === color ? 'bg-cyan-950/30 text-cyan-200' : 'text-gray-300',
                      )}
                    >
                      {color}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {layers.map(layer => {
                  const assignedColors = new Set(
                    layer.colorPalette.flatMap(color => {
                      const normalized = normalizeHexColor(color);
                      return normalized ? [normalized] : [];
                    }),
                  );
                  return (
                    <tr key={layer.elementIndex} className="h-[27px] bg-[#07131c] hover:bg-[#0b1822]">
                      <th
                        scope="row"
                        className="sticky left-0 z-10 max-w-[180px] border-b border-r border-[#2b3741] bg-inherit px-2.5 text-left text-[11px] font-medium text-gray-100 last:border-b-0"
                        title={layer.label}
                      >
                        <span className="block truncate">{layer.label}</span>
                      </th>
                      {normalizedPalette.map(({ color }) => {
                        const assigned = assignedColors.has(color);
                        const capacityReached = !assigned && assignedColors.size >= 5;
                        const inputDisabled = disabled || capacityReached;
                        return (
                          <td
                            key={color}
                            className={classNames(
                              'border-b border-r border-[#2b3741] text-center last:border-r-0',
                              normalizedSelectedColor === color && 'bg-cyan-950/20',
                            )}
                          >
                            <button
                              type="button"
                              role="checkbox"
                              aria-checked={assigned}
                              disabled={inputDisabled}
                              onClick={() => onToggleLayerColor(layer.elementIndex, color, !assigned)}
                              aria-label={`${assigned ? 'Remove' : 'Assign'} ${color} ${assigned ? 'from' : 'to'} ${layer.label}`}
                              title={capacityReached ? `${layer.label} already has the maximum of 5 colors` : undefined}
                              className={classNames(
                                'inline-flex h-[18px] w-[18px] items-center justify-center rounded-[2px] border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-35',
                                assigned
                                  ? 'border-cyan-300 bg-cyan-400 text-[#03121a]'
                                  : 'border-transparent bg-transparent text-transparent hover:border-gray-600 hover:bg-white/5',
                              )}
                            >
                              {assigned ? <Check className="h-3.5 w-3.5" strokeWidth={3} aria-hidden="true" /> : null}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex h-[72px] items-center justify-center rounded-[4px] border border-dashed border-[#33414d] text-xs text-gray-500">
            {normalizedPalette.length === 0 ? 'No palette colors to assign' : 'No layers available'}
          </div>
        )}
      </div>
    </section>
  );
}
