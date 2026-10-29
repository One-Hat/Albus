import React from 'react';
import type { Source, VerdictLabel } from '@contracts/types';

interface SourceBadgeProps {
  source: Source;
  size?: 'xs' | 'sm' | 'md';
}

export const SourceBadge: React.FC<SourceBadgeProps> = ({
  source,
  size = 'xs',
}) => {
  const isMock = source === 'MOCK';
  const sizeClasses =
    size === 'md'
      ? 'px-2.5 py-0.5 text-xs font-bold'
      : size === 'sm'
      ? 'px-2 py-0.5 text-[11px] font-semibold'
      : 'px-1.5 py-0.2 text-[10px] font-semibold';

  return (
    <span
      data-testid={`source-badge-${source}`}
      className={`inline-flex items-center gap-1 rounded border font-mono uppercase tracking-wider select-none ${sizeClasses} ${
        isMock
          ? 'bg-amber-500/15 text-amber-300 border-amber-500/40'
          : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
      }`}
      title={
        isMock
          ? 'MOCK DATA: Synthetic fixture — never interpret as real financial signal'
          : 'REAL DATA: Verified SEC EDGAR & Groq open-source model output'
      }
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          isMock ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'
        }`}
      />
      {source}
    </span>
  );
};

interface TaggedNumberProps {
  value: React.ReactNode;
  source: Source;
  className?: string;
  badgeSize?: 'xs' | 'sm';
}

export const TaggedNumber: React.FC<TaggedNumberProps> = ({
  value,
  source,
  className = '',
  badgeSize = 'xs',
}) => {
  return (
    <span
      data-source={source}
      className={`inline-flex items-center gap-1.5 font-mono ${className}`}
    >
      <span>{value}</span>
      <SourceBadge source={source} size={badgeSize} />
    </span>
  );
};

export const VerdictBadge: React.FC<{ label: VerdictLabel; size?: 'sm' | 'lg' }> = ({
  label,
  size = 'sm',
}) => {
  const styles: Record<VerdictLabel, string> = {
    pass: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40',
    weak: 'bg-amber-500/15 text-amber-300 border-amber-500/40',
    reject: 'bg-rose-500/15 text-rose-300 border-rose-500/40',
  };

  const sizeClass =
    size === 'lg'
      ? 'px-3 py-1 text-sm font-bold tracking-widest'
      : 'px-2 py-0.5 text-xs font-semibold tracking-wider';

  return (
    <span
      className={`inline-flex items-center rounded border font-mono uppercase ${sizeClass} ${styles[label]}`}
    >
      {label}
    </span>
  );
};
