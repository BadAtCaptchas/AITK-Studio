'use client';

import { useSearchParams } from '@/navigation';
import { TrainingFormContent } from './TrainingFormContent';

export default function TrainingForm() {
  const params = useSearchParams();
  return (
    <TrainingFormContent
      key={
        params.get('migrateId')
          ? 'migrate:' + params.get('migrateId')
          : params.get('id') || (params.get('cloneId') ? 'clone:' + params.get('cloneId') : 'new')
      }
    />
  );
}
