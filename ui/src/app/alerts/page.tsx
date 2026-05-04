'use client';

import { Bell, Check, Clock, ToggleLeft, ToggleRight } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import useAlerts from '@/hooks/useAlerts';

function severityClass(severity: string) {
  if (severity === 'critical') return 'text-red-400';
  if (severity === 'warning') return 'text-yellow-400';
  return 'text-blue-400';
}

export default function AlertsPage() {
  const { events, rules, updateEvent, updateRule } = useAlerts(5000);

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-3">
          <Bell className="h-5 w-5 text-blue-400" />
          <h1 className="text-base text-white">Alerts</h1>
          <span className="text-xs text-gray-500">Inbox and rules</span>
        </div>
      </TopBar>
      <MainContent>
        <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
          <section className="overflow-hidden border border-white/10 bg-black">
            <div className="border-b border-white/10 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-white">Inbox</div>
            <div className="divide-y divide-white/5">
              {events.map(event => (
                <div key={event.id} className="grid gap-3 px-4 py-3 md:grid-cols-[120px_1fr_220px]">
                  <div>
                    <div className={`text-xs font-semibold uppercase ${severityClass(event.severity)}`}>{event.severity}</div>
                    <div className="mt-1 text-xs text-gray-500">{event.status}</div>
                  </div>
                  <div className="min-w-0">
                    <div className="truncate text-sm text-white">{event.title}</div>
                    <div className="mt-1 truncate text-xs text-gray-400">{event.message}</div>
                    <div className="mt-2 text-[11px] text-gray-600">
                      {event.resource_type}:{event.resource_id}
                    </div>
                  </div>
                  <div className="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => void updateEvent(event.id, { status: 'acknowledged' })}
                      className="inline-flex items-center gap-1 rounded border border-white/10 px-2 py-1 text-xs text-gray-300 hover:bg-white/5"
                    >
                      <Check className="h-3.5 w-3.5" />
                      Ack
                    </button>
                    <button
                      type="button"
                      onClick={() => void updateEvent(event.id, { status: 'resolved' })}
                      className="inline-flex items-center gap-1 rounded border border-white/10 px-2 py-1 text-xs text-gray-300 hover:bg-white/5"
                    >
                      <Clock className="h-3.5 w-3.5" />
                      Resolve
                    </button>
                  </div>
                </div>
              ))}
              {events.length === 0 && <div className="px-4 py-10 text-center text-sm text-gray-500">No alerts.</div>}
            </div>
          </section>

          <section className="border border-white/10 bg-black">
            <div className="border-b border-white/10 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-white">Rules</div>
            <div className="divide-y divide-white/5">
              {rules.map(rule => (
                <div key={rule.id} className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => void updateRule(rule.id, { enabled: !rule.enabled })}
                      className={rule.enabled ? 'text-green-400' : 'text-gray-600'}
                      title={rule.enabled ? 'Disable rule' : 'Enable rule'}
                    >
                      {rule.enabled ? <ToggleRight className="h-5 w-5" /> : <ToggleLeft className="h-5 w-5" />}
                    </button>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm text-white">{rule.name}</div>
                      <div className="mt-1 text-xs text-gray-500">
                        {rule.metric} {rule.operator} {rule.threshold}
                      </div>
                    </div>
                    <div className={`text-xs uppercase ${severityClass(rule.severity)}`}>{rule.severity}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </MainContent>
    </>
  );
}
