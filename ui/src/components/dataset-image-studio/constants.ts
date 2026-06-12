export const MIN_BOX_SPAN = 8;
export const MAX_HISTORY = 50;
export const THUMB_WINDOW = 11;
export const CLICK_DRAG_TOLERANCE = 4;
export const BOX_COLORS = ['#22D3EE', '#F59E0B', '#A3E635', '#FB7185', '#818CF8', '#34D399'];
export const HEX_COLOR_PATTERN = /^#[0-9A-F]{6}$/;
export const DEFAULT_OPENROUTER_BOX_MODEL = 'x-ai/grok-4.3';
export const DEFAULT_OLLAMA_VISION_MODEL = 'qwen3.5:35b';
export const AUTO_BOX_PROVIDERS = [
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'remote_ollama', label: 'Remote Ollama' },
] as const;
export const OPENROUTER_BOX_MODELS = [
  { value: DEFAULT_OPENROUTER_BOX_MODEL, label: 'x-ai/grok-4.3' },
];
export const OLLAMA_VISION_MODELS = [
  { value: 'qwen3.5:122b', label: 'qwen3.5:122b (best quality, high VRAM)' },
  { value: DEFAULT_OLLAMA_VISION_MODEL, label: 'qwen3.5:35b (recommended)' },
  { value: 'qwen3.5:27b', label: 'qwen3.5:27b (backup)' },
  { value: 'qwen3.5:9b', label: 'qwen3.5:9b (small backup)' },
  { value: 'gemma4:31b', label: 'gemma4:31b' },
  { value: 'gemma4:26b', label: 'gemma4:26b' },
];
