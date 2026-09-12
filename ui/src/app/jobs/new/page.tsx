'use client';

import { useSearchParams } from 'next/navigation';
import { TrainingFormContent } from './TrainingFormContent';

export default function TrainingForm() {
  const params = useSearchParams();
  return <TrainingFormContent key={params.get('id') || (params.get('cloneId') ? 'clone:' + params.get('cloneId') : 'new')} />;
}
