'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { SystemMetricSample } from '@/types';

export default function useSystemTelemetry(range: '1h' | '6h' | '24h' | 'all' = '6h', reloadInterval = 5000) {
  const [samples, setSamples] = useState<SystemMetricSample[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    apiClient
      .get('/api/system/telemetry', { params: { range } })
      .then(res => res.data)
      .then(data => {
        setSamples(data.samples ?? []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching telemetry:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refresh();
    if (!reloadInterval) return;
    const id = window.setInterval(refresh, reloadInterval);
    return () => window.clearInterval(id);
  }, [range, reloadInterval]);

  return { samples, status, refresh };
}
