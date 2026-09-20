'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { QwenImageControls } from '@/components/generate/QwenImageControls';
import { transparentQwenPrompt } from '@/domain/qwenImage';
import { apiClient } from '@/utils/api';
import { startJob, stopJob } from '@/utils/jobs';
import useGPUInfo from '@/hooks/useGPUInfo';
import usePollLoop from '@/hooks/usePollLoop';
import { useModelArchs } from '@/extensions/modelArchs';
import { quantizationOptions } from '@/domain/modelOptions';
import LoraBrowserModal, { LoraPick } from '@/components/generate/LoraBrowserModal';
import { readEngineFrames, payloadToFloat32, latentToImage, parsePreview, PreviewInfo } from '@/utils/engineStream';
import { SelectInput, TextInput, TextAreaInput, NumberInput, Checkbox } from '@/components/formInputs';

const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
type EngineJob = { id: string; name: string; status: string; gpu_ids: string };
type Lora = LoraPick & { strength: number; enabled: boolean };
type Result = { relpath: string; kind: string; seed?: number };
type Phase = { type: 'idle' } | { type: 'starting' } | { type: 'generating'; requestID: string | null };
function readJobs(value: unknown): EngineJob[] {
  const list = record(value) && Array.isArray(value.jobs) ? value.jobs : Array.isArray(value) ? value : [];
  return list.flatMap((j: unknown) =>
    record(j) &&
    typeof j.id === 'string' &&
    typeof j.name === 'string' &&
    typeof j.status === 'string' &&
    typeof j.gpu_ids === 'string'
      ? [{ id: j.id, name: j.name, status: j.status, gpu_ids: j.gpu_ids }]
      : [],
  );
}
function TextResult({ url }: { url: string }) {
  const [text, setText] = useState('Loading…');
  useEffect(() => {
    const c = new AbortController();
    void fetch(url, { signal: c.signal })
      .then(r => {
        if (!r.ok) throw new Error('Could not read output');
        return r.text();
      })
      .then(setText)
      .catch(() => {
        if (!c.signal.aborted) setText('Could not read output.');
      });
    return () => c.abort();
  }, [url]);
  return <pre className="whitespace-pre-wrap text-sm">{text}</pre>;
}
export default function LiveGeneratePage() {
  const { archs, groupedModelOptions, errors } = useModelArchs();
  const { gpuList } = useGPUInfo();
  const [gpu, setGpu] = useState('0');
  const [jobs, setJobs] = useState<EngineJob[]>([]);
  const [jobID, setJobID] = useState('');
  const initialJobsLoaded = useRef(false);
  const [arch, setArch] = useState('flux');
  const [model, setModel] = useState<Record<string, unknown>>({
    arch: 'flux',
    name_or_path: 'black-forest-labs/FLUX.1-dev',
    dtype: 'bf16',
    quantize: true,
    qtype: 'qfloat8',
  });
  const [prompt, setPrompt] = useState('');
  const [negative, setNegative] = useState('');
  const [width, setWidth] = useState(1024),
    [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(25),
    [guidance, setGuidance] = useState(4);
  const [seed, setSeed] = useState(-1),
    [frames, setFrames] = useState(1),
    [fps, setFps] = useState(16),
    [duration, setDuration] = useState(120);
  const [controls, setControls] = useState<string[]>([]);
  const [loras, setLoras] = useState<Lora[]>([]),
    [loraOpen, setLoraOpen] = useState(false);
  const [loraMode, setLoraMode] = useState('hook');
  const [phase, setPhase] = useState<Phase>({ type: 'idle' });
  const [ready, setReady] = useState(false),
    [message, setMessage] = useState('Start or select an engine.');
  const [error, setError] = useState(''),
    [progress, setProgress] = useState(0);
  const [results, setResults] = useState<Result[]>([]);
  const [modelJSON, setModelJSON] = useState('');
  const canvas = useRef<HTMLCanvasElement>(null),
    streamAbort = useRef<AbortController | null>(null);
  const preview = useRef<PreviewInfo | null>(null);
  const selected = archs.find(item => item.name === arch);
  const endpoint = useCallback(
    (route: string) => `/api/inference/${route}${route.includes('?') ? '&' : '?'}job_id=${encodeURIComponent(jobID)}`,
    [jobID],
  );
  const refreshJobs = useCallback(async () => {
    const data: unknown = (
      await apiClient.get<unknown>('/api/jobs', { params: { job_type: 'inference', local_only: '1' } })
    ).data;
    const next = readJobs(data);
    setJobs(next);
    if (!initialJobsLoaded.current) {
      initialJobsLoaded.current = true;
      setJobID(next.find(job => job.status === 'running')?.id || next[0]?.id || '');
    }
  }, []);
  useEffect(() => {
    void refreshJobs().catch(() => setError('Could not list engines.'));
  }, [refreshJobs]);
  usePollLoop(
    async signal => {
      if (!jobID) return;
      try {
        const response = await fetch(endpoint('health'), { signal });
        setReady(response.ok);
      } catch {
        if (!signal.aborted) setReady(false);
      }
    },
    2000,
    [jobID],
  );
  useEffect(() => () => streamAbort.current?.abort(), []);
  const changeArch = (name: string) => {
    setArch(name);
    setControls([]);
    const definition = archs.find(item => item.name === name);
    const next: Record<string, unknown> = { arch: name, dtype: 'bf16' };
    for (const [key, pair] of Object.entries(definition?.defaults || {})) {
      const field = key.replace('config.process[0].model.', '');
      if (key.startsWith('config.process[0].model.') && !field.includes('.') && Array.isArray(pair))
        next[field] = pair[0];
    }
    setModel(next);
    setModelJSON('');
    const sample: unknown = definition?.defaults?.['config.process[0].sample']?.[0];
    if (record(sample)) {
      if (typeof sample.sample_steps === 'number') setSteps(sample.sample_steps);
      if (typeof sample.guidance_scale === 'number') setGuidance(sample.guidance_scale);
      if (typeof sample.duration === 'number') setDuration(sample.duration);
    }
    const presetGuidance: unknown = definition?.defaults?.['config.process[0].sample.guidance_scale']?.[0];
    if (typeof presetGuidance === 'number') setGuidance(presetGuidance);
    setFrames(definition?.isVideoModel ? 33 : 1);
  };
  const start = async () => {
    setPhase({ type: 'starting' });
    setError('');
    setMessage('Starting engine…');
    try {
      let id = jobID;
      if (!id) {
        const name = `inference_${Date.now()}`;
        const response: unknown = (
          await apiClient.post<unknown>('/api/jobs', {
            name,
            job_type: 'inference',
            gpu_ids: gpu,
            worker_id: 'local',
            job_config: {
              job: 'extension',
              config: { name, process: [{ type: 'InferenceEngine', device: 'cuda', dtype: 'bf16', engine: {} }] },
            },
          })
        ).data;
        if (!record(response) || typeof response.id !== 'string') throw new Error('Invalid engine job response');
        id = response.id;
        setJobID(id);
      }
      await startJob(id);
      await refreshJobs();
      setMessage('Engine is starting.');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start engine');
    } finally {
      setPhase({ type: 'idle' });
    }
  };
  const consume = async (response: Response, controller: AbortController) => {
    if (!response.ok) {
      const body: unknown = await response.json().catch(() => null);
      throw new Error(
        record(body) && typeof body.error === 'string' ? body.error : `Engine returned HTTP ${response.status}`,
      );
    }
    setPhase({ type: 'generating', requestID: response.headers.get('x-request-id') });
    await readEngineFrames(
      response,
      ({ header, payload }) => {
        if (typeof header.request_id === 'string') setPhase({ type: 'generating', requestID: header.request_id });
        if (header.preview) preview.current = parsePreview(header.preview);
        if (header.type === 'status' && typeof header.message === 'string') setMessage(header.message);
        if (header.type === 'progress' && typeof header.step === 'number' && typeof header.total === 'number')
          setProgress(header.total ? header.step / header.total : 0);
        if (header.type === 'latent') {
          const image = latentToImage(header, payloadToFloat32(header, payload), preview.current);
          if (image && canvas.current) {
            canvas.current.width = image.width;
            canvas.current.height = image.height;
            const context = canvas.current.getContext('2d');
            const pixels = new Uint8ClampedArray(image.frameData[0]);
            context?.putImageData(new ImageData(pixels, image.width, image.height), 0, 0);
          }
        }
        if (header.type === 'result' && typeof header.relpath === 'string' && typeof header.kind === 'string') {
          const result = {
            relpath: header.relpath.replaceAll('\\', '/'),
            kind: header.kind,
            seed: typeof header.seed === 'number' ? header.seed : undefined,
          };
          setResults(current => (current.some(r => r.relpath === result.relpath) ? current : [result, ...current]));
        }
        if (header.type === 'error')
          setError(typeof header.message === 'string' ? header.message : 'Generation failed');
        if (header.type === 'end')
          setMessage(header.status === 'done' ? 'Generation complete.' : String(header.status || 'Finished'));
      },
      controller.signal,
    );
  };
  const generate = async (attachID?: string) => {
    const controller = new AbortController();
    streamAbort.current = controller;
    setPhase({ type: 'generating', requestID: attachID || null });
    setError('');
    setProgress(0);
    try {
      let overrides: Record<string, unknown> = {};
      if (modelJSON.trim()) {
        const value: unknown = JSON.parse(modelJSON);
        if (!record(value)) throw new Error('Model options must be a JSON object');
        overrides = value;
      }
      const response = await fetch(endpoint(attachID ? `stream/${encodeURIComponent(attachID)}` : 'generate'), {
        method: attachID ? 'GET' : 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: attachID
          ? undefined
          : JSON.stringify({
              model: {
                ...model,
                ...overrides,
                arch,
                lora_mode: loraMode,
                loras: loras.filter(l => l.enabled).map(({ path, strength }) => ({ path, strength })),
              },
              sample: {
                prompt,
                negative_prompt: negative,
                width,
                height,
                num_inference_steps: steps,
                guidance_scale: guidance,
                seed,
                num_frames: frames,
                fps,
                duration,
                ...(arch === 'qwen_image_2' ? { ctrl_imgs: controls } : {}),
                ctrl_img: controls[0],
                ctrl_img_1: controls[0],
                ctrl_img_2: controls[1],
                ctrl_img_3: controls[2],
              },
              stream: { latents: 'raw', every_n_steps: 1, max_frames: 4 },
            }),
      });
      await consume(response, controller);
    } catch (e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      streamAbort.current = null;
      setPhase({ type: 'idle' });
    }
  };
  const reconnect = async () => {
    try {
      const data: unknown = (await apiClient.get<unknown>(endpoint('queue'))).data;
      if (!record(data)) throw new Error('Invalid queue response');
      const candidates = [
        ...(Array.isArray(data.queued) ? data.queued : []),
        ...(Array.isArray(data.recent) ? data.recent : []),
      ];
      const recent =
        candidates.find((item: unknown) => record(item) && ['running', 'queued'].includes(String(item.status))) ||
        candidates[0];
      if (record(recent) && typeof recent.request_id === 'string') await generate(recent.request_id);
      else setMessage('No recent generation to reconnect.');
    } catch {
      setError('Could not reconnect to generation.');
    }
  };
  const busy = phase.type !== 'idle';
  return (
    <main className="mx-auto max-w-7xl p-5 lg:p-8 space-y-5">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Live inference</h1>
          <p className="text-sm text-gray-400">Keep a model loaded, combine adapters, and watch generation progress.</p>
        </div>
        <Link href="/generate" className="operator-button">
          Native / ComfyUI generation
        </Link>
      </header>
      <div className="grid gap-3 sm:grid-cols-4 items-end">
        <SelectInput
          label="Engine"
          value={jobID}
          onChange={value => {
            setJobID(value);
            setReady(false);
            setResults([]);
            setControls([]);
          }}
          options={[
            { value: '', label: 'New engine' },
            ...jobs.map(j => ({ value: j.id, label: `${j.name} (${j.status})` })),
          ]}
          disabled={busy}
        />
        <SelectInput
          label="GPU for new engine"
          value={gpu}
          onChange={setGpu}
          options={gpuList.map(g => ({ value: String(g.index), label: `GPU ${g.index}: ${g.name}` }))}
          disabled={!!jobID || busy}
        />
        <button className="operator-button" disabled={busy || ready} onClick={start}>
          {phase.type === 'starting' ? 'Starting…' : 'Start engine'}
        </button>
        <button
          className="operator-button"
          disabled={!jobID || busy}
          onClick={async () => {
            try {
              await stopJob(jobID);
              setReady(false);
              await refreshJobs();
            } catch {
              setError('Could not stop engine.');
            }
          }}
        >
          Stop engine
        </button>
      </div>
      <p role="status" className="text-sm text-gray-400">
        {ready ? 'Engine ready. ' : ''}
        {message}
      </p>
      {(error || errors.length > 0) && (
        <p role="alert" className="text-red-400">
          {error || errors.join('; ')}
        </p>
      )}
      <div className="grid lg:grid-cols-2 gap-6">
        <section className="space-y-4">
          <SelectInput label="Model architecture" value={arch} onChange={changeArch} options={groupedModelOptions} />
          <TextInput
            label="Model path"
            value={String(model.name_or_path || '')}
            onChange={v => setModel(m => ({ ...m, name_or_path: v }))}
          />
          <div className="grid grid-cols-2 gap-3">
            <SelectInput
              label="Quantization"
              value={model.quantize ? String(model.qtype || 'qfloat8') : ''}
              options={quantizationOptions}
              onChange={v => setModel(m => ({ ...m, quantize: !!v, qtype: v || 'qfloat8' }))}
            />
            <SelectInput
              label="Precision"
              value={String(model.dtype || 'bf16')}
              options={['bf16', 'fp16', 'fp32'].map(v => ({ value: v, label: v }))}
              onChange={v => setModel(m => ({ ...m, dtype: v }))}
            />
          </div>
          <div className="flex flex-wrap gap-4">
            <Checkbox
              label="Quantize text encoder"
              checked={model.quantize_te === true}
              onChange={v => setModel(m => ({ ...m, quantize_te: v }))}
            />
            <Checkbox
              label="Low VRAM"
              checked={model.low_vram === true}
              onChange={v => setModel(m => ({ ...m, low_vram: v }))}
            />
            <Checkbox
              label="Layer offloading"
              checked={model.layer_offloading === true}
              onChange={v => setModel(m => ({ ...m, layer_offloading: v }))}
            />
          </div>
          <details>
            <summary className="cursor-pointer text-sm">Additional model options (JSON)</summary>
            <TextAreaInput
              label="Model options"
              value={modelJSON}
              onChange={setModelJSON}
              placeholder={'{"model_kwargs": {}}'}
            />
          </details>
          {arch === 'qwen_image_2' && <QwenImageControls
            options={record(model.model_kwargs) ? model.model_kwargs : {}}
            onOption={(key, value) => setModel(current => ({ ...current, model_kwargs: { ...(record(current.model_kwargs) ? current.model_kwargs : {}), [key]: value } }))}
            onPreset={(width, height, steps) => { setWidth(width); setHeight(height); setSteps(steps); }}
            onTransparentPrompt={() => setPrompt(transparentQwenPrompt(prompt))}
          />}
          <TextAreaInput label="Prompt / instruction" value={prompt} onChange={setPrompt} />
          <TextAreaInput label="Negative prompt" value={negative} onChange={setNegative} />
          <div className="grid grid-cols-3 gap-3">
            <NumberInput label="Width" value={width} min={1} onChange={v => setWidth(v ?? 1024)} />
            <NumberInput label="Height" value={height} min={1} onChange={v => setHeight(v ?? 1024)} />
            <NumberInput label="Seed" value={seed} min={-1} onChange={v => setSeed(v ?? -1)} />
            <NumberInput label="Steps" value={steps} min={1} onChange={v => setSteps(v ?? 25)} />
            <NumberInput label="Guidance" value={guidance} min={0} onChange={v => setGuidance(v ?? 4)} />
            <NumberInput label="Duration (seconds)" value={duration} min={1} onChange={v => setDuration(v ?? 120)} />
            <NumberInput label="Frames" value={frames} min={1} onChange={v => setFrames(v ?? 1)} />
            <NumberInput label="FPS" value={fps} min={1} onChange={v => setFps(v ?? 16)} />
          </div>
          <label className="block text-sm">
            Reference image, audio, or video
            <input
              type="file"
              multiple
              accept="image/*,audio/*,video/*"
              disabled={!ready || busy}
              className="block mt-2"
              onChange={async e => {
                try {
                  const uploaded: string[] = [];
                  const files = Array.from(e.target.files || []);
                  const maxReferences = arch === 'qwen_image_2' ? 10 : 3;
                  if (files.length > maxReferences) throw new Error(`Choose at most ${maxReferences} references.`);
                  for (const file of files) {
                    const response = await fetch(endpoint(`assets?name=${encodeURIComponent(file.name)}`), {
                      method: 'POST',
                      body: file,
                    });
                    const data: unknown = await response.json();
                    if (!response.ok || !record(data) || typeof data.path !== 'string')
                      throw new Error('Media upload failed');
                    uploaded.push(data.path);
                  }
                  setControls(uploaded);
                } catch (e) {
                  setError(e instanceof Error ? e.message : 'Media upload failed');
                }
              }}
            />
          </label>
          {controls.length > 0 && <ol className="space-y-1 text-sm">
            {controls.map((path, index) => <li key={`${path}-${index}`} className="flex items-center gap-2">
              <span className="truncate">{index + 1}. {path.split(/[\\/]/).pop()}</span>
              <button type="button" disabled={busy || index === 0} onClick={() => setControls(current => { const next = [...current]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; return next; })}>Up</button>
              <button type="button" disabled={busy} onClick={() => setControls(current => current.filter((_, i) => i !== index))}>Remove</button>
            </li>)}
          </ol>}
          {controls.length > 0 && (
            <button className="text-sm text-gray-400" onClick={() => setControls([])}>
              Clear {controls.length} reference file(s)
            </button>
          )}
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">LoRA / LoKr stack</h2>
            <button className="operator-button" onClick={() => setLoraOpen(true)}>
              Add adapter
            </button>
          </div>
          {loras.map((l, i) => (
            <div key={l.path} className="flex items-center gap-2 rounded bg-gray-900 p-2">
              <Checkbox
                label={l.name}
                checked={l.enabled}
                onChange={v =>
                  setLoras(current => current.map((item, j) => (j === i ? { ...item, enabled: v } : item)))
                }
              />
              <input
                aria-label={`Strength for ${l.name}`}
                type="number"
                step="0.05"
                min="-4"
                max="4"
                value={l.strength}
                className="w-20 bg-gray-800 rounded p-1"
                onChange={e =>
                  setLoras(current =>
                    current.map((item, j) => (j === i ? { ...item, strength: Number(e.target.value) } : item)),
                  )
                }
              />
              <button onClick={() => setLoras(current => current.filter((_, j) => j !== i))}>Remove</button>
            </div>
          ))}
          <SelectInput
            label="Adapter mode"
            value={loraMode}
            onChange={setLoraMode}
            options={[
              { value: 'hook', label: 'Sidechain (switch without reloading)' },
              { value: 'merge', label: 'Merge into model' },
            ]}
          />
          {selected?.modelNotes && <p className="text-xs text-gray-400">{selected.modelNotes.summary}</p>}
          <div className="flex flex-wrap gap-2">
            <button
              className="operator-button"
              disabled={!ready || busy || !prompt.trim()}
              onClick={() => void generate()}
            >
              Generate
            </button>
            <button className="operator-button" disabled={!ready || busy} onClick={reconnect}>
              Reconnect / latest result
            </button>
            <button
              className="operator-button"
              disabled={phase.type !== 'generating' || !phase.requestID}
              onClick={async () => {
                try {
                  if (phase.type === 'generating' && phase.requestID)
                    await apiClient.post(endpoint(`cancel/${phase.requestID}`), {});
                } catch {
                  setError('Could not cancel generation.');
                }
              }}
            >
              Cancel generation
            </button>
            <button
              className="operator-button"
              disabled={!ready || busy}
              onClick={async () => {
                try {
                  await apiClient.post(endpoint('unload'), {});
                  setMessage('Model unloaded.');
                } catch {
                  setError('Could not unload model.');
                }
              }}
            >
              Unload model
            </button>
          </div>
        </section>
        <section className="space-y-4">
          <progress aria-label="Generation progress" value={progress} max={1} className="w-full" />
          <canvas
            ref={canvas}
            className="w-full max-h-[60vh] object-contain rounded bg-gray-950"
            aria-label="Latent generation preview"
          />
          <h2 className="font-semibold">Outputs</h2>
          {results.map(result => {
            const url = endpoint(`outputs/${result.relpath.split('/').map(encodeURIComponent).join('/')}`);
            return (
              <article key={result.relpath} className="rounded border border-gray-800 p-3 space-y-2">
                {result.kind === 'text' ? (
                  <TextResult url={url} />
                ) : result.kind === 'audio' ? (
                  <audio controls src={url} className="w-full" />
                ) : result.kind === 'video' ? (
                  <video controls src={url} className="w-full" />
                ) : (
                  <img src={url} alt="Generated output" className="w-full" />
                )}
                <a href={url} download className="text-sm text-blue-300">
                  Download · seed {result.seed ?? '?'}
                </a>
              </article>
            );
          })}
        </section>
      </div>
      <LoraBrowserModal
        isOpen={loraOpen}
        onClose={() => setLoraOpen(false)}
        onPick={pick => {
          setLoras(current =>
            current.some(l => l.path === pick.path) ? current : [...current, { ...pick, strength: 1, enabled: true }],
          );
          if (pick.triggerWords?.length) setMessage(`Adapter trigger words: ${pick.triggerWords.join(', ')}`);
        }}
      />
    </main>
  );
}
