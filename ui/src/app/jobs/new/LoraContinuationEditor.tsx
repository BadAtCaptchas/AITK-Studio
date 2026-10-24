import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import Link from '@/navigation/Link';
import { apiClient } from '@/utils/api';
import { uploadLoraFile } from '@/utils/streamedUploads';
import type { JobConfig, SelectOption, NetworkConfig } from '@/types';
import { defaultDatasetConfig } from './jobConfig';
import { continuationSourceKey, parseLoraInspection, preservationModes, type LoraInspection, type PreservationMode } from '@/domain/loraContinuation';

type Props = {
  config: JobConfig;
  setConfig: (value: JobConfig | ((previous: JobConfig) => JobConfig)) => void;
  datasets: SelectOption[];
  disabled?: boolean;
  workerID: string;
};
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const control = 'w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-950';
const labels: Record<PreservationMode, string> = { none: 'Off', replay: 'Curated replay', teacher: 'Frozen-source teacher (experimental)', replay_teacher: 'Replay + teacher (experimental)' };

export default function LoraContinuationEditor({ config, setConfig, datasets, workerID, disabled = false }: Props) {
  const process = config.config.process[0];
  const source = process.network?.continuation;
  const enabled = source !== undefined;
  const [report, setReport] = useState<LoraInspection | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [library, setLibrary] = useState<{ path: string; label: string }[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const sourceKey = continuationSourceKey(process, workerID);
  const currentSourceKey = useRef(sourceKey);
  useLayoutEffect(() => {
    currentSourceKey.current = sourceKey;
    requestId.current++;
    setReport(null);
    setBusy(false);
    setMessage('');
  }, [sourceKey]);
  useEffect(() => () => { requestId.current++; }, []);
  useEffect(() => {
    requestId.current++;
    let current = true;
    setLibrary([]);
    setReport(null);
    setBusy(false);
    void apiClient.get<unknown>('/api/generate/loras', { params: { worker_id: workerID } }).then(({ data }) => {
      if (!current || !record(data) || !Array.isArray(data.loras)) return;
      setLibrary(data.loras.filter(record).flatMap(item => typeof item.path === 'string' && typeof item.label === 'string' ? [{ path: item.path, label: item.label }] : []));
    }).catch(() => undefined);
    return () => { current = false; };
  }, [workerID]);
  const update = (change: (process: JobConfig['config']['process'][number] & { network: NetworkConfig }) => void, expectedKey?: string) => setConfig(previous => {
    if (expectedKey !== undefined && (currentSourceKey.current !== expectedKey ||
        continuationSourceKey(previous.config.process[0], workerID) !== expectedKey)) return previous;
    const next = structuredClone(previous);
    const process = next.config.process[0];
    if (!process.network) return previous;
    const editable = { ...process, network: process.network };
    change(editable);
    next.config.process[0] = editable;
    return next;
  });
  const setPath = (value: string, expectedKey?: string) => {
    requestId.current++;
    setBusy(false);
    setReport(null);
    setMessage('Inspect this source before saving the job.');
    update(p => {
      p.network.pretrained_lora_path = value;
      if (p.network.continuation) p.network.continuation.source_sha256 = '';
    }, expectedKey);
  };
  const inspect = async () => {
    const id = ++requestId.current;
    update(p => { if (p.network.continuation) p.network.continuation.source_sha256 = ''; }, sourceKey);
    setReport(null);
    setBusy(true);
    setMessage('Inspecting tensors and Krea 2 targets…');
    try {
      const { data } = await apiClient.post<unknown>('/api/training/lora/inspect', { path: process.network?.pretrained_lora_path, scale: source?.source_scale });
      const result = parseLoraInspection(data);
      if (requestId.current !== id) return;
      setReport(result);
      update(p => {
        if (!p.network.continuation) return;
        p.network.continuation.source_sha256 = result.sha256;
        p.network.linear = result.ranks[0];
        p.network.linear_alpha = result.ranks[0];
      }, sourceKey);
      setMessage(`${result.modules.length} targets verified; source rank${result.ranks.length > 1 ? 's' : ''}: ${result.ranks.join(', ')}. Rank, scaling and targets come from the file.`);
    } catch (error: unknown) {
      if (requestId.current !== id) return;
      const response = record(error) && record(error.response) ? error.response.data : undefined;
      setMessage(record(response) && typeof response.error === 'string' ? response.error : error instanceof Error ? error.message : 'Inspection failed.');
    } finally { if (requestId.current === id) setBusy(false); }
  };
  const upload = async (file: File) => {
    const id = ++requestId.current;
    setBusy(true);
    setMessage(workerID === 'local' ? 'Uploading LoRA locally…' : 'Uploading LoRA to the selected worker…');
    try {
      const result = await uploadLoraFile(file, { workerID });
      if (requestId.current !== id) return;
      const data: unknown = result.data;
      if (!record(data) || !record(data.lora) || typeof data.lora.path !== 'string') throw new Error('Upload did not return a source file.');
      setPath(data.lora.path, sourceKey);
    } catch (error) { if (requestId.current === id) setMessage(error instanceof Error ? error.message : 'Upload failed.'); }
    finally { if (requestId.current === id) setBusy(false); if (fileInput.current) fileInput.current.value = ''; }
  };
  const setMode = (mode: PreservationMode) => update(p => {
    p.train.source_preservation = mode === 'none' ? { mode } : { mode, new_batches: 3, teacher_weight: 1 };
    if (mode === 'none') p.datasets = p.datasets.filter(dataset => dataset.role !== 'source_preservation');
  });
  const preservation = process.train.source_preservation;
  const selectedDataset = process.datasets.find(dataset => dataset.role === 'source_preservation');
  const remoteSource = process.network?.pretrained_lora_path?.startsWith('aitk-lora://remote/');
  return <fieldset disabled={disabled || busy} className="studio-disclosure space-y-3 p-4 disabled:opacity-70">
    <label className="flex items-center gap-2 text-sm font-medium">
      <input type="checkbox" checked={enabled} onChange={event => update(p => {
        if (event.target.checked) {
          p.network.type = 'lora';
          p.network.continuation = { version: 1, source_sha256: '', training_base: p.model.arch === 'krea2:turbo' ? 'turbo' : 'raw', deployment_base: 'turbo' };
          delete p.train.start_step;
          p.train.source_preservation = { mode: 'none' };
        } else {
          delete p.network.continuation;
          delete p.network.pretrained_lora_path;
          delete p.train.source_preservation;
          p.datasets = p.datasets.filter(dataset => dataset.role !== 'source_preservation');
        }
      })} /> Further train this LoRA
    </label>
    <p className="text-xs text-gray-500">Start a new run from downloaded weights and export one complete updated LoRA. Original optimizer state is not restored.</p>
    {enabled && <>
      <label className="block text-xs">Source from library
        <select className={control} value={process.network?.pretrained_lora_path || ''} onChange={event => setPath(event.target.value)}>
          <option value="">Select a LoRA</option>
          {library.map(item => <option key={item.path} value={item.path}>{item.label}</option>)}
          {process.network?.pretrained_lora_path && !library.some(item => item.path === process.network?.pretrained_lora_path) && <option value={process.network.pretrained_lora_path}>{process.network.pretrained_lora_path}</option>}
        </select>
      </label>
      <label className="block text-xs">Source path<input aria-label="Continuation source path" className={control} value={process.network?.pretrained_lora_path || ''} onChange={event => setPath(event.target.value)} /></label>
      <div className="flex gap-3">
        <button type="button" className="operator-button" onClick={() => fileInput.current?.click()}>{workerID === 'local' ? 'Upload LoRA locally' : 'Upload LoRA to worker'}</button>
        <input ref={fileInput} type="file" accept=".safetensors" className="hidden" onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); }} />
        <button type="button" className="operator-button" disabled={!process.network?.pretrained_lora_path} onClick={() => void inspect()}>Inspect LoRA</button>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <label className="text-xs">Continuation training base<select className={control} value={source.training_base} onChange={event => update(p => { if (p.network.continuation) p.network.continuation.training_base = event.target.value === 'turbo' ? 'turbo' : 'raw'; })}>
          <option value="raw">Krea 2 Raw</option><option value="turbo">Krea 2 Turbo</option>
        </select></label>
        <label className="text-xs">Deployment base<select className={control} value={source.deployment_base} onChange={event => update(p => { if (p.network.continuation) p.network.continuation.deployment_base = event.target.value === 'raw' ? 'raw' : 'turbo'; })}>
          <option value="turbo">Krea 2 Turbo</option><option value="raw">Krea 2 Raw</option>
        </select></label>
        <label className="text-xs">Declared scale, if metadata is missing<input aria-label="Declared source scale" className={control} type="number" min="0.000001" step="any" placeholder="Auto" value={source.source_scale ?? ''} onChange={event => {
          setReport(null);
          update(p => { if (p.network.continuation) { p.network.continuation.source_scale = event.target.value === '' ? undefined : Number(event.target.value); p.network.continuation.source_sha256 = ''; } });
        }} /></label>
      </div>
      <p className="text-xs text-gray-500">Match the selected model checkpoint. A Turbo model-card listing does not establish which base originally trained the adapter.</p>
      <p role="status" className="text-xs text-gray-400">{message || (source.source_sha256 ? `Inspected source: ${source.source_sha256.slice(0, 12)}…` : 'Inspection is required.')}</p>
      {report?.declared_training_base && <p className="text-xs">File declares training base: {report.declared_training_base}</p>}
      <label className="block text-xs">Preserve source behavior<select className={control} value={preservation?.mode || 'none'} onChange={event => { const mode = preservationModes.find(value => value === event.target.value); if (mode) setMode(mode); }}>
        {preservationModes.map(mode => <option key={mode} value={mode}>{labels[mode]}</option>)}
      </select></label>
      <p className="text-xs text-gray-500">Curated replay is the recommended starting point. Teacher guidance is experimental; neither guarantees unchanged behavior.</p>
      {source.source_sha256 && (remoteSource ? <p className="text-xs text-gray-400">Open Generate on the source worker to create preservation candidates, then select them in sample review.</p> : <Link className="text-sm text-blue-400 underline" href={`/generate?model_ref=${encodeURIComponent(process.network?.pretrained_lora_path || '')}&continuation_base=${source.deployment_base}${source.source_scale === undefined ? '' : `&source_scale=${source.source_scale}`}`}>Generate preservation candidates with the source LoRA</Link>)}
      {preservation && preservation.mode !== 'none' && <>
        <label className="block text-xs">Curated preservation dataset<select className={control} value={selectedDataset?.folder_path || ''} onChange={event => update(p => {
          p.datasets = p.datasets.filter(dataset => dataset.role !== 'source_preservation');
          if (event.target.value) p.datasets.push({ ...structuredClone(defaultDatasetConfig), folder_path: event.target.value, role: 'source_preservation', caption_dropout_rate: 0, caption_ext: 'txt' });
        })}><option value="">Select a dataset created from reviewed samples</option>{datasets.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs">New batches per preservation batch<input className={control} type="number" min="1" max="100" value={preservation.new_batches} onChange={event => update(p => { if (p.train.source_preservation && p.train.source_preservation.mode !== 'none') p.train.source_preservation.new_batches = Number(event.target.value); })} /></label>
          {preservation.mode !== 'replay' && <label className="text-xs">Teacher loss weight<input className={control} type="number" min="0.000001" step="0.1" value={preservation.teacher_weight} onChange={event => update(p => { if (p.train.source_preservation && p.train.source_preservation.mode !== 'none') p.train.source_preservation.teacher_weight = Number(event.target.value); })} /></label>}
        </div>
      </>}
    </>}
  </fieldset>;
}
