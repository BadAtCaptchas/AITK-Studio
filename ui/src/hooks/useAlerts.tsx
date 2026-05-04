'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { AlertEvent, AlertRule } from '@/types';

export default function useAlerts(reloadInterval: number | null = 5000) {
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    Promise.all([apiClient.get('/api/alerts'), apiClient.get('/api/alerts/rules')])
      .then(([eventsRes, rulesRes]) => {
        setEvents(eventsRes.data.events ?? []);
        setRules(rulesRes.data.rules ?? []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching alerts:', error);
        setStatus('error');
      });
  };

  const updateEvent = async (id: string, payload: Record<string, unknown>) => {
    await apiClient.patch(`/api/alerts/${id}`, payload);
    refresh();
  };

  const updateRule = async (id: string, payload: Record<string, unknown>) => {
    await apiClient.patch('/api/alerts/rules', { id, ...payload });
    refresh();
  };

  useEffect(() => {
    refresh();
    if (!reloadInterval) return;
    const id = window.setInterval(refresh, reloadInterval);
    return () => window.clearInterval(id);
  }, [reloadInterval]);

  return { events, rules, status, refresh, updateEvent, updateRule };
}
