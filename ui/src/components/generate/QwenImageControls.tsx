import { useState } from 'react';
import { Checkbox, TextInput } from '@/components/formInputs';
import { QWEN_IMAGE_PRESETS } from '@/domain/qwenImage';
import { uploadTemporaryMediaFile } from '@/utils/streamedUploads';

type OptionsProps = {
  options: Record<string, unknown>;
  onOption: (key: string, value: boolean) => void;
  onPreset: (width: number, height: number, steps: number) => void;
  onTransparentPrompt?: () => void;
};
export function QwenImageControls({ options, onOption, onPreset, onTransparentPrompt }: OptionsProps) {
  return <div className="space-y-3 border border-gray-800 p-3">
    <label className="block text-sm">Qwen Image 2.1 quality preset
      <select className="mt-1 block w-full bg-gray-900 p-2" value="" onChange={event => {
        const preset = QWEN_IMAGE_PRESETS[Number(event.target.value)];
        if (preset) onPreset(preset.width, preset.height, preset.steps);
      }}>
        <option value="" disabled>Choose a preset</option>
        {QWEN_IMAGE_PRESETS.map((preset, i) => <option key={preset.label} value={i}>{preset.label}</option>)}
      </select>
    </label>
    <p className="text-xs text-gray-400">2K presets use 40 steps and more memory. Guidance stays at your chosen value; AI Toolkit defaults to 3, Diffusers to 1.</p>
    <Checkbox label="Transparent output (RGBA)" checked={options.rgba === true} onChange={value => onOption('rgba', value)} />
    {onTransparentPrompt && <button type="button" className="text-sm text-blue-300" onClick={() => { onOption('rgba', true); onTransparentPrompt(); }}>Add transparency instructions to prompt</button>}
    <Checkbox label="Reuse reference and text cache during sampling" checked={options.use_kv_cache !== false} onChange={value => onOption('use_kv_cache', value)} />
    <Checkbox label="Expand prompts with the official Qwen rewriter" checked={options.rewrite_prompt === true} onChange={value => onOption('rewrite_prompt', value)} />
    <p className="text-xs text-gray-400">Prompt expansion downloads a separate 9B model when first used, runs on CPU, and may take several minutes. References select the editing rewriter. Your chosen canvas size is preserved.</p>
  </div>;
}

async function uploadReference(file: File): Promise<string> {
  const response = await uploadTemporaryMediaFile(file);
  const data: unknown = response.data;
  if (!data || typeof data !== 'object' || !('files' in data) || !Array.isArray(data.files) || typeof data.files[0] !== 'string') throw new Error('Invalid media upload response');
  return data.files[0];
}
export function QwenReferenceInputs({ paths, onChange }: { paths: string[]; onChange: (paths: string[]) => void }) {
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  return <fieldset className="space-y-2" disabled={uploading}>
    <legend className="text-sm">Reference images ({paths.length}/10)</legend>
    <p className="text-xs text-gray-400">References are passed in this order as image1, image2, and so on. Annotated images and separate masks can be included as references.</p>
    {paths.map((path, i) => <div key={i} className="flex items-end gap-2">
      <TextInput label={`Reference ${i + 1}`} value={path} onChange={value => onChange(paths.map((item, index) => index === i ? value : item))} />
      <button type="button" aria-label={`Move reference ${i + 1} up`} disabled={i === 0} onClick={() => {
        const next = [...paths]; [next[i - 1], next[i]] = [next[i], next[i - 1]]; onChange(next);
      }}>Up</button>
      <button type="button" aria-label={`Remove reference ${i + 1}`} onClick={() => onChange(paths.filter((_, index) => index !== i))}>Remove</button>
    </div>)}
    <button type="button" className="text-sm text-blue-300" disabled={paths.length >= 10} onClick={() => onChange([...paths, ''])}>Add reference path</button>
    <input aria-label="Upload Qwen reference images" type="file" multiple accept="image/*" disabled={paths.length >= 10} onChange={async event => {
      const files = Array.from(event.target.files || []); event.target.value = '';
      if (files.length + paths.length > 10) { setError('Choose at most 10 references in total.'); return; }
      setUploading(true); setError('');
      try { const uploaded: string[] = []; for (const file of files) uploaded.push(await uploadReference(file)); onChange([...paths, ...uploaded]); }
      catch (error) { setError(error instanceof Error ? error.message : 'Upload failed'); }
      finally { setUploading(false); }
    }} />
    {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
  </fieldset>;
}
