export const UPDATER_STATUS_EVENT = 'aitk-updater-status';

export function broadcastUpdaterStatus(status: unknown) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(UPDATER_STATUS_EVENT, { detail: status }));
}
