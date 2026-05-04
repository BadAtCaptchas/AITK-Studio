import { db } from './db';

const DEFAULT_ALERT_RULES = [
  { id: 'gpu-temperature-high', name: 'GPU temperature high', metric: 'temperature', operator: '>=', threshold: 85, duration_seconds: 0, severity: 'critical', config: { scope: 'gpu', unit: 'C' } },
  { id: 'vram-utilization-high', name: 'VRAM utilization high', metric: 'vram_utilization', operator: '>=', threshold: 95, duration_seconds: 0, severity: 'warning', config: { scope: 'gpu', unit: '%' } },
  { id: 'ram-utilization-high', name: 'RAM utilization high', metric: 'ram_utilization', operator: '>=', threshold: 90, duration_seconds: 0, severity: 'warning', config: { scope: 'ram', unit: '%' } },
  { id: 'power-limit-high', name: 'GPU power near limit', metric: 'power_draw_percent', operator: '>=', threshold: 95, duration_seconds: 0, severity: 'warning', config: { scope: 'gpu', unit: '%' } },
  { id: 'job-no-progress', name: 'Running job has no recent progress', metric: 'job_no_progress_minutes', operator: '>=', threshold: 15, duration_seconds: 0, severity: 'warning', config: {} },
  { id: 'job-failed', name: 'Job failed', metric: 'job_failed', operator: '>=', threshold: 1, duration_seconds: 0, severity: 'critical', config: {} },
  { id: 'missing-artifact', name: 'Model or dataset reference missing', metric: 'missing_artifact', operator: '>=', threshold: 1, duration_seconds: 0, severity: 'warning', config: {} },
  { id: 'evaluation-failed', name: 'Evaluation failed', metric: 'evaluation_failed', operator: '>=', threshold: 1, duration_seconds: 0, severity: 'warning', config: {} },
];

function safeJsonParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function compare(value: number, operator: string, threshold: number) {
  if (operator === '>') return value > threshold;
  if (operator === '<=') return value <= threshold;
  if (operator === '<') return value < threshold;
  if (operator === '=') return value === threshold;
  return value >= threshold;
}

export async function ensureDefaultAlertRules() {
  const existing = new Set((await db.alerts.listRules()).map(rule => rule.id));
  await Promise.all(
    DEFAULT_ALERT_RULES.filter(rule => !existing.has(rule.id)).map(rule =>
      db.alerts.upsertRule({
        ...rule,
        config: JSON.stringify(rule.config),
      }),
    ),
  );
}

async function createEvent(input: {
  ruleID: string;
  title: string;
  message: string;
  severity: string;
  resourceType: string;
  resourceID: string;
  fingerprint: string;
  metadata?: Record<string, unknown>;
}) {
  await db.alerts.upsertEvent({
    rule_id: input.ruleID,
    title: input.title,
    message: input.message,
    severity: input.severity,
    status: 'active',
    resource_type: input.resourceType,
    resource_id: input.resourceID,
    fingerprint: input.fingerprint,
    metadata: JSON.stringify(input.metadata ?? {}),
  });
}

export async function evaluateAlerts() {
  await ensureDefaultAlertRules();
  const activeFingerprints = new Set<string>();
  const rules = (await db.alerts.listRules()).filter(rule => rule.enabled);
  const ruleByID = new Map(rules.map(rule => [rule.id, rule]));

  const telemetry = await db.systemMetrics.list({ since: new Date(Date.now() - 10 * 60 * 1000), limit: 10000 });
  const latest = new Map<string, (typeof telemetry)[number]>();
  for (const sample of telemetry) {
    latest.set(`${sample.scope}:${sample.device_id}:${sample.metric}`, sample);
  }

  for (const rule of rules) {
    const config = safeJsonParse<{ scope?: string }>(rule.config, {});
    if (!['temperature', 'vram_utilization', 'ram_utilization'].includes(rule.metric)) continue;
    for (const sample of latest.values()) {
      if (sample.metric !== rule.metric) continue;
      if (config.scope && sample.scope !== config.scope) continue;
      if (!compare(sample.value, rule.operator, rule.threshold)) continue;
      const fingerprint = `${rule.id}:${sample.scope}:${sample.device_id}`;
      activeFingerprints.add(fingerprint);
      await createEvent({
        ruleID: rule.id,
        title: rule.name,
        message: `${sample.scope.toUpperCase()} ${sample.device_id} ${sample.metric} is ${sample.value.toFixed(1)}${sample.unit}.`,
        severity: rule.severity,
        resourceType: sample.scope,
        resourceID: sample.device_id,
        fingerprint,
        metadata: { value: sample.value, unit: sample.unit },
      });
    }
  }

  const failedJobRule = ruleByID.get('job-failed');
  if (failedJobRule) {
    const failedJobs = await db.jobs.list({ status: 'error' });
    for (const job of failedJobs) {
      const fingerprint = `${failedJobRule.id}:${job.id}`;
      activeFingerprints.add(fingerprint);
      await createEvent({
        ruleID: failedJobRule.id,
        title: `Job failed: ${job.name}`,
        message: job.info || 'Training job is in the error state.',
        severity: failedJobRule.severity,
        resourceType: 'job',
        resourceID: job.id,
        fingerprint,
      });
    }
  }

  const noProgressRule = ruleByID.get('job-no-progress');
  if (noProgressRule) {
    const runningJobs = await db.jobs.list({ status: 'running' });
    for (const job of runningJobs) {
      const updatedAt = new Date(job.updated_at).getTime();
      const minutes = (Date.now() - updatedAt) / 60000;
      if (!compare(minutes, noProgressRule.operator, noProgressRule.threshold)) continue;
      const fingerprint = `${noProgressRule.id}:${job.id}`;
      activeFingerprints.add(fingerprint);
      await createEvent({
        ruleID: noProgressRule.id,
        title: `No recent progress: ${job.name}`,
        message: `Job has not updated for ${minutes.toFixed(1)} minutes.`,
        severity: noProgressRule.severity,
        resourceType: 'job',
        resourceID: job.id,
        fingerprint,
        metadata: { minutes },
      });
    }
  }

  const missingRule = ruleByID.get('missing-artifact');
  if (missingRule) {
    const missing = await db.modelArtifacts.list({ exists: false });
    for (const artifact of missing) {
      const fingerprint = `${missingRule.id}:${artifact.id}`;
      activeFingerprints.add(fingerprint);
      await createEvent({
        ruleID: missingRule.id,
        title: `Missing artifact: ${artifact.name}`,
        message: artifact.path,
        severity: missingRule.severity,
        resourceType: 'model_artifact',
        resourceID: artifact.id,
        fingerprint,
      });
    }
  }

  const evalRule = ruleByID.get('evaluation-failed');
  if (evalRule) {
    const failedRuns = await db.evaluations.listRuns({ status: 'failed', limit: 100 });
    for (const run of failedRuns) {
      const fingerprint = `${evalRule.id}:${run.id}`;
      activeFingerprints.add(fingerprint);
      await createEvent({
        ruleID: evalRule.id,
        title: `Evaluation failed: ${run.name}`,
        message: run.error || 'Evaluation runner failed.',
        severity: evalRule.severity,
        resourceType: 'evaluation',
        resourceID: run.id,
        fingerprint,
      });
    }
  }

  const currentEvents = await db.alerts.listEvents({ status: 'active', limit: 1000 });
  await Promise.all(
    currentEvents
      .filter(event => event.rule_id && ruleByID.has(event.rule_id) && !activeFingerprints.has(event.fingerprint))
      .map(event => db.alerts.resolveByFingerprint(event.fingerprint)),
  );
}
