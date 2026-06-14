import {
  arrayToBox,
  cloneIdeogramData,
  normalizeIdeogramColorPalette,
  parseIdeogramCaption,
  serializeIdeogramCaption,
  type IdeogramElementType,
  type NormalizedBox,
} from './ideogramCaption';

export type IdeogramQualityPreset = 'Quality' | 'Default' | 'Turbo';

export type IdeogramWorkflowElement = {
  type: IdeogramElementType;
  bbox: [number, number, number, number];
  desc: string;
  text?: string;
  color_palette?: string[];
};

export type IdeogramWorkflowModels = {
  diffusion: string;
  unconditional: string;
  clip: string;
  vae: string;
};

export type IdeogramWorkflowLora = {
  loraName: string;
  strengthModel: number;
  strengthClip: number;
  toolkitPath?: string;
};

export type IdeogramWorkflowState = {
  highLevelDescription: string;
  style: {
    aesthetics: string;
    lighting: string;
    photo: string;
    medium: string;
    colorPalette: string[];
  };
  background: string;
  elements: IdeogramWorkflowElement[];
  aspectRatio: string;
  megapixels: number;
  seed: number;
  qualityPreset: IdeogramQualityPreset;
  steps: number;
  samplerName: string;
  guiderCfg: number;
  overrideCfg: number;
  overrideStartPercent: number;
  overrideEndPercent: number;
  filenamePrefix: string;
  models: IdeogramWorkflowModels;
  loras: IdeogramWorkflowLora[];
};

export type IdeogramImportResult = {
  state: IdeogramWorkflowState;
  workflow: Record<string, unknown>;
  warnings: string[];
};

export type IdeogramRequiredModel = {
  id: string;
  label: string;
  classType: string;
  inputName: string;
  value: string;
};

export const IDEOGRAM4_DEFAULT_MODELS: IdeogramWorkflowModels = {
  diffusion: 'ideogram4_fp8_scaled.safetensors',
  unconditional: 'ideogram4_unconditional_fp8_scaled.safetensors',
  clip: 'qwen3vl_8b_fp8_scaled.safetensors',
  vae: 'flux2-vae.safetensors',
};

export const IDEOGRAM4_QUALITY_PRESETS: Record<IdeogramQualityPreset, { steps: number; mu: number; std: number; preset_id: string }> = {
  Quality: { steps: 48, mu: 0.0, std: 1.5, preset_id: 'V4_QUALITY_48' },
  Default: { steps: 20, mu: 0.0, std: 1.75, preset_id: 'V4_DEFAULT_20' },
  Turbo: { steps: 12, mu: 0.5, std: 1.75, preset_id: 'V4_TURBO_12' },
};

const QUALITY_OPTIONS: IdeogramQualityPreset[] = ['Quality', 'Default', 'Turbo'];

export const IDEOGRAM_ASPECT_RATIOS = [
  '1:1 (Square)',
  '2:3 (Portrait Photo)',
  '3:2 (Landscape Photo)',
  '4:5 (Portrait)',
  '9:16 (Vertical)',
  '16:9 (Widescreen)',
];

const ASPECT_VALUE_RE = /^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/;

export const IDEOGRAM4_REQUIRED_NODE_CLASSES = [
  'ResolutionSelector',
  'SaveImage',
  'VAELoader',
  'ConditioningZeroOut',
  'EmptyFlux2LatentImage',
  'SamplerCustomAdvanced',
  'VAEDecode',
  'KSamplerSelect',
  'Ideogram4Scheduler',
  'RandomNoise',
  'UNETLoader',
  'CLIPTextEncode',
  'CLIPLoader',
  'PrimitiveInt',
  'ComfyMathExpression',
  'ComfyNumberConvert',
  'JsonExtractString',
  'StringReplace',
  'CustomCombo',
  'CFGOverride',
  'DualModelGuider',
];

export const DEFAULT_IDEOGRAM_WORKFLOW_STATE: IdeogramWorkflowState = {
  highLevelDescription:
    'A dynamic skateboarder jumping in an urban environment, bold typography saying COMFY, cinematic lighting, photorealistic, high detail.',
  style: {
    aesthetics: 'documentary, low-budget found-footage realism, cinematic streetwear poster',
    lighting: 'dim lighting, strong contrast, wet street reflections, practical street lights',
    photo: 'phone camera, snapshot, subtle lens distortion, visible sensor noise, handheld',
    medium: 'product photograph',
    colorPalette: ['#0EA5E9', '#0891B2', '#F59E0B', '#EF4444', '#E5E7EB'],
  },
  background: 'A rain-slick downtown street at night with soft bokeh traffic lights and dark glass buildings.',
  elements: [
    {
      type: 'text',
      bbox: [82, 174, 382, 823],
      text: 'COMFY',
      desc: 'Large distressed white block typography spelling COMFY across the top half of the poster.',
      color_palette: ['#E5E7EB', '#94A3B8'],
    },
    {
      type: 'obj',
      bbox: [447, 175, 904, 824],
      desc: 'A hooded skateboarder mid-jump above the wet street, centered below the text with dramatic motion.',
      color_palette: ['#111827', '#1F2937', '#F59E0B'],
    },
  ],
  aspectRatio: '2:3 (Portrait Photo)',
  megapixels: 1,
  seed: 672389204,
  qualityPreset: 'Default',
  steps: 20,
  samplerName: 'euler',
  guiderCfg: 4,
  overrideCfg: 3,
  overrideStartPercent: 0.7,
  overrideEndPercent: 1,
  filenamePrefix: 'Ideogram_4.0',
  models: { ...IDEOGRAM4_DEFAULT_MODELS },
  loras: [],
};

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function finiteNumber(value: unknown, fallback: number) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function finiteInt(value: unknown, fallback: number) {
  return Math.round(finiteNumber(value, fallback));
}

function positiveFiniteInt(value: unknown, fallback: number) {
  return Math.max(1, finiteInt(value, fallback));
}

function cleanText(value: unknown, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function cleanLora(raw: unknown): IdeogramWorkflowLora | null {
  if (!isRecord(raw)) return null;
  const loraName = cleanText(raw.loraName || raw.lora_name || raw.name).trim();
  const toolkitPath = cleanText(raw.toolkitPath || raw.toolkit_path).trim();
  return {
    loraName,
    strengthModel: finiteNumber(raw.strengthModel ?? raw.strength_model, 1),
    strengthClip: finiteNumber(raw.strengthClip ?? raw.strength_clip, 1),
    ...(toolkitPath ? { toolkitPath } : {}),
  };
}

function cleanLoras(value: unknown): IdeogramWorkflowLora[] {
  if (!Array.isArray(value)) return [];
  return value.map(cleanLora).filter((item): item is IdeogramWorkflowLora => item !== null);
}

function cleanElement(raw: unknown): IdeogramWorkflowElement | null {
  if (!isRecord(raw)) return null;
  const box = arrayToBox(raw.bbox);
  if (!box) return null;
  const type: IdeogramElementType = raw.type === 'text' ? 'text' : 'obj';
  const text = cleanText(raw.text).trim();
  const desc = cleanText(raw.desc || raw.description || text, type === 'text' ? text : 'Visible object').trim();
  const palette = normalizeIdeogramColorPalette(raw.color_palette || raw.colorPalette || raw.palette);
  return {
    type,
    bbox: [box.y1, box.x1, box.y2, box.x2],
    ...(type === 'text' ? { text } : {}),
    desc: desc || (type === 'text' ? text : 'Visible object'),
    ...(palette.length > 0 ? { color_palette: palette } : {}),
  };
}

export function cloneIdeogramWorkflowState(state: IdeogramWorkflowState = DEFAULT_IDEOGRAM_WORKFLOW_STATE): IdeogramWorkflowState {
  const next = deepClone(state);
  next.loras = cleanLoras(next.loras);
  return next;
}

export function updateIdeogramWorkflowElementBox(
  state: IdeogramWorkflowState,
  elementIndex: number,
  box: NormalizedBox,
): IdeogramWorkflowState {
  const next = cloneIdeogramWorkflowState(state);
  const element = next.elements[elementIndex];
  if (!element) return next;
  element.bbox = [box.y1, box.x1, box.y2, box.x2];
  return next;
}

export function stateToIdeogramData(state: IdeogramWorkflowState) {
  return {
    high_level_description: state.highLevelDescription,
    style_description: {
      aesthetics: state.style.aesthetics,
      lighting: state.style.lighting,
      photo: state.style.photo,
      medium: state.style.medium,
      color_palette: normalizeIdeogramColorPalette(state.style.colorPalette),
    },
    compositional_deconstruction: {
      background: state.background,
      elements: state.elements.map(element => ({
        type: element.type,
        bbox: element.bbox,
        ...(element.type === 'text' ? { text: element.text || '' } : {}),
        desc: element.desc,
        ...(element.color_palette?.length ? { color_palette: normalizeIdeogramColorPalette(element.color_palette) } : {}),
      })),
    },
  };
}

export function serializeIdeogramWorkflowPrompt(state: IdeogramWorkflowState) {
  return serializeIdeogramCaption(stateToIdeogramData(state));
}

export function ideogramDataToState(
  data: Record<string, any>,
  baseState: IdeogramWorkflowState = DEFAULT_IDEOGRAM_WORKFLOW_STATE,
): IdeogramWorkflowState {
  const next = cloneIdeogramWorkflowState(baseState);
  next.highLevelDescription = cleanText(data.high_level_description, next.highLevelDescription);

  const style = isRecord(data.style_description) ? data.style_description : {};
  next.style = {
    aesthetics: cleanText(style.aesthetics, next.style.aesthetics),
    lighting: cleanText(style.lighting, next.style.lighting),
    photo: cleanText(style.photo, next.style.photo),
    medium: cleanText(style.medium, next.style.medium),
    colorPalette: normalizeIdeogramColorPalette(style.color_palette || next.style.colorPalette),
  };

  const composition = isRecord(data.compositional_deconstruction) ? data.compositional_deconstruction : {};
  next.background = cleanText(composition.background, next.background);
  if (Array.isArray(composition.elements)) {
    const elements = composition.elements.map(cleanElement).filter((item): item is IdeogramWorkflowElement => item !== null);
    if (elements.length > 0) next.elements = elements;
  }
  return next;
}

function presetJsonString() {
  return JSON.stringify(
    Object.fromEntries(
      Object.entries(IDEOGRAM4_QUALITY_PRESETS).map(([name, preset]) => [
        name,
        {
          num_steps: preset.steps,
          mu: preset.mu,
          std: preset.std,
          preset_id: preset.preset_id,
        },
      ]),
    ),
    null,
    2,
  );
}

function qualityIndex(preset: IdeogramQualityPreset) {
  return Math.max(0, QUALITY_OPTIONS.indexOf(preset)) + 1;
}

export function parseIdeogramAspectRatio(value: string) {
  const match = value.match(ASPECT_VALUE_RE);
  const width = match ? Number(match[1]) : 2;
  const height = match ? Number(match[2]) : 3;
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { width: 2, height: 3, css: '2 / 3', ratio: 2 / 3 };
  }
  return { width, height, css: `${width} / ${height}`, ratio: width / height };
}

export function closestIdeogramAspectRatio(width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return DEFAULT_IDEOGRAM_WORKFLOW_STATE.aspectRatio;
  }
  const target = width / height;
  return IDEOGRAM_ASPECT_RATIOS.reduce((best, option) => {
    const bestDelta = Math.abs(parseIdeogramAspectRatio(best).ratio - target);
    const nextDelta = Math.abs(parseIdeogramAspectRatio(option).ratio - target);
    return nextDelta < bestDelta ? option : best;
  }, IDEOGRAM_ASPECT_RATIOS[0]);
}

export function requiredIdeogramModels(state: IdeogramWorkflowState = DEFAULT_IDEOGRAM_WORKFLOW_STATE): IdeogramRequiredModel[] {
  const loras = cleanLoras(state.loras).filter(lora => lora.loraName.trim());
  return [
    {
      id: 'unet-main',
      label: 'UNET',
      classType: 'UNETLoader',
      inputName: 'unet_name',
      value: state.models.diffusion,
    },
    {
      id: 'unet-unconditional',
      label: 'UNET unconditional',
      classType: 'UNETLoader',
      inputName: 'unet_name',
      value: state.models.unconditional,
    },
    {
      id: 'clip',
      label: 'CLIP',
      classType: 'CLIPLoader',
      inputName: 'clip_name',
      value: state.models.clip,
    },
    {
      id: 'vae',
      label: 'VAE',
      classType: 'VAELoader',
      inputName: 'vae_name',
      value: state.models.vae,
    },
    ...loras.map((lora, index) => ({
      id: `lora-${index}`,
      label: `LoRA ${index + 1}`,
      classType: 'LoraLoader',
      inputName: 'lora_name',
      value: lora.loraName,
    })),
  ];
}

export function buildIdeogramComfyWorkflow(state: IdeogramWorkflowState = DEFAULT_IDEOGRAM_WORKFLOW_STATE) {
  const qualityPreset = QUALITY_OPTIONS.includes(state.qualityPreset) ? state.qualityPreset : 'Default';
  const selectedPreset = IDEOGRAM4_QUALITY_PRESETS[qualityPreset] || IDEOGRAM4_QUALITY_PRESETS.Default;
  const steps = positiveFiniteInt(state.steps, selectedPreset.steps);
  const promptText = serializeIdeogramWorkflowPrompt(state);
  const loras = cleanLoras(state.loras).filter(lora => lora.loraName.trim());

  const workflow: Record<string, any> = {
    '37': {
      inputs: {
        aspect_ratio: state.aspectRatio,
        megapixels: state.megapixels,
      },
      class_type: 'ResolutionSelector',
      _meta: { title: 'Resolution Selector' },
    },
    '158': {
      inputs: {
        filename_prefix: state.filenamePrefix || 'Ideogram_4.0',
        images: ['98:13', 0],
      },
      class_type: 'SaveImage',
      _meta: { title: 'Save Image' },
    },
    '98:9': {
      inputs: { vae_name: state.models.vae },
      class_type: 'VAELoader',
      _meta: { title: 'Load VAE' },
    },
    '98:10': {
      inputs: { conditioning: ['98:24', 0] },
      class_type: 'ConditioningZeroOut',
      _meta: { title: 'ConditioningZeroOut' },
    },
    '98:11': {
      inputs: {
        width: ['98:31', 1],
        height: ['98:32', 1],
        batch_size: 1,
      },
      class_type: 'EmptyFlux2LatentImage',
      _meta: { title: 'Empty Flux 2 Latent' },
    },
    '98:12': {
      inputs: {
        noise: ['98:18', 0],
        guider: ['98:155', 0],
        sampler: ['98:16', 0],
        sigmas: ['98:17', 0],
        latent_image: ['98:11', 0],
      },
      class_type: 'SamplerCustomAdvanced',
      _meta: { title: 'SamplerCustomAdvanced' },
    },
    '98:13': {
      inputs: {
        samples: ['98:12', 0],
        vae: ['98:9', 0],
      },
      class_type: 'VAEDecode',
      _meta: { title: 'VAE Decode' },
    },
    '98:16': {
      inputs: { sampler_name: state.samplerName || 'euler' },
      class_type: 'KSamplerSelect',
      _meta: { title: 'KSamplerSelect' },
    },
    '98:17': {
      inputs: {
        steps: ['98:151', 1],
        width: ['98:31', 1],
        height: ['98:32', 1],
        mu: ['98:144', 0],
        std: ['98:146', 0],
      },
      class_type: 'Ideogram4Scheduler',
      _meta: { title: 'Ideogram 4 Scheduler' },
    },
    '98:18': {
      inputs: { noise_seed: finiteInt(state.seed, DEFAULT_IDEOGRAM_WORKFLOW_STATE.seed) },
      class_type: 'RandomNoise',
      _meta: { title: 'RandomNoise' },
    },
    '98:23': {
      inputs: {
        unet_name: state.models.diffusion,
        weight_dtype: 'default',
      },
      class_type: 'UNETLoader',
      _meta: { title: 'Load Diffusion Model' },
    },
    '98:24': {
      inputs: {
        text: promptText,
        clip: ['98:14', 0],
      },
      class_type: 'CLIPTextEncode',
      _meta: { title: 'CLIP Text Encode (Positive Prompt)' },
    },
    '98:14': {
      inputs: {
        clip_name: state.models.clip,
        type: 'ideogram4',
        device: 'default',
      },
      class_type: 'CLIPLoader',
      _meta: { title: 'Load CLIP' },
    },
    '98:27': {
      inputs: { value: ['37', 0] },
      class_type: 'PrimitiveInt',
      _meta: { title: 'Int (Width)' },
    },
    '98:28': {
      inputs: { value: ['37', 1] },
      class_type: 'PrimitiveInt',
      _meta: { title: 'Int (Height)' },
    },
    '98:31': {
      inputs: {
        expression: 'max(((a + 15) // 16) * 16, 256)',
        'values.a': ['98:27', 0],
      },
      class_type: 'ComfyMathExpression',
      _meta: { title: 'Math Expression' },
    },
    '98:32': {
      inputs: {
        expression: 'max(((a + 15) // 16) * 16, 256)',
        'values.a': ['98:28', 0],
      },
      class_type: 'ComfyMathExpression',
      _meta: { title: 'Math Expression' },
    },
    '98:144': {
      inputs: { value: ['98:145', 0] },
      class_type: 'ComfyNumberConvert',
      _meta: { title: 'Number Convert' },
    },
    '98:145': {
      inputs: {
        json_string: ['98:148', 0],
        key: 'mu',
      },
      class_type: 'JsonExtractString',
      _meta: { title: 'Extract Text from JSON' },
    },
    '98:146': {
      inputs: { value: ['98:150', 0] },
      class_type: 'ComfyNumberConvert',
      _meta: { title: 'Number Convert' },
    },
    '98:147': {
      inputs: {
        json_string: presetJsonString(),
        key: ['98:156', 0],
      },
      class_type: 'JsonExtractString',
      _meta: { title: 'Extract Text from JSON' },
    },
    '98:148': {
      inputs: {
        string: ['98:147', 0],
        find: "'",
        replace: '"',
      },
      class_type: 'StringReplace',
      _meta: { title: 'Replace Text' },
    },
    '98:149': {
      inputs: {
        json_string: ['98:148', 0],
        key: 'num_steps',
      },
      class_type: 'JsonExtractString',
      _meta: { title: 'Extract Text from JSON' },
    },
    '98:150': {
      inputs: {
        json_string: ['98:148', 0],
        key: 'std',
      },
      class_type: 'JsonExtractString',
      _meta: { title: 'Extract Text from JSON' },
    },
    '98:151': {
      inputs: { value: steps },
      class_type: 'ComfyNumberConvert',
      _meta: { title: 'Number Convert' },
    },
    '98:154': {
      inputs: {
        unet_name: state.models.unconditional,
        weight_dtype: 'default',
      },
      class_type: 'UNETLoader',
      _meta: { title: 'Load Diffusion Model' },
    },
    '98:156': {
      inputs: {
        choice: qualityPreset,
        index: qualityIndex(qualityPreset) - 1,
        option1: 'Quality',
        option2: 'Default',
        option3: 'Turbo',
        option4: '',
      },
      class_type: 'CustomCombo',
      _meta: { title: 'Custom Combo' },
    },
    '98:157': {
      inputs: {
        cfg: finiteNumber(state.overrideCfg, 3),
        start_percent: finiteNumber(state.overrideStartPercent, 0.7),
        end_percent: finiteNumber(state.overrideEndPercent, 1),
        model: ['98:23', 0],
      },
      class_type: 'CFGOverride',
      _meta: { title: 'CFG Override' },
    },
    '98:155': {
      inputs: {
        cfg: finiteNumber(state.guiderCfg, 4),
        model: ['98:157', 0],
        positive: ['98:24', 0],
        model_negative: ['98:154', 0],
        negative: ['98:10', 0],
      },
      class_type: 'DualModelGuider',
      _meta: { title: 'Dual Model CFG Guider' },
    },
  };

  let positiveModelLink: [string, number] = ['98:23', 0];
  let negativeModelLink: [string, number] = ['98:154', 0];
  let clipLink: [string, number] = ['98:14', 0];

  loras.forEach((lora, index) => {
    const positiveId = `98:${177 + index * 2}`;
    const negativeId = `98:${178 + index * 2}`;
    const strengthModel = finiteNumber(lora.strengthModel, 1);
    const strengthClip = finiteNumber(lora.strengthClip, 1);

    workflow[positiveId] = {
      inputs: {
        lora_name: lora.loraName,
        strength_model: strengthModel,
        strength_clip: strengthClip,
        model: positiveModelLink,
        clip: clipLink,
      },
      class_type: 'LoraLoader',
      _meta: { title: `Load LoRA ${index + 1} (Positive)` },
    };
    workflow[negativeId] = {
      inputs: {
        lora_name: lora.loraName,
        strength_model: strengthModel,
        strength_clip: strengthClip,
        model: negativeModelLink,
        clip: clipLink,
      },
      class_type: 'LoraLoader',
      _meta: { title: `Load LoRA ${index + 1} (Negative)` },
    };

    positiveModelLink = [positiveId, 0];
    negativeModelLink = [negativeId, 0];
    clipLink = [negativeId, 1];
  });

  workflow['98:157'].inputs.model = positiveModelLink;
  workflow['98:155'].inputs.model_negative = negativeModelLink;
  workflow['98:24'].inputs.clip = clipLink;

  return workflow;
}

function getNode(workflow: Record<string, any>, id: string) {
  const node = workflow[id];
  return isRecord(node) ? node : null;
}

function input(workflow: Record<string, any>, id: string, key: string) {
  const inputs = getNode(workflow, id)?.inputs;
  return isRecord(inputs) ? inputs[key] : undefined;
}

function findNodesByClass(workflow: Record<string, any>, classType: string) {
  return Object.entries(workflow)
    .filter(([, node]) => isRecord(node) && node.class_type === classType)
    .map(([id, node]) => ({ id, node }));
}

function linkNodeId(value: unknown) {
  return Array.isArray(value) && typeof value[0] === 'string' ? value[0] : null;
}

function traceLoraModelChain(workflow: Record<string, any>, terminalLink: unknown, baseNodeId: string) {
  const chain: Array<{ id: string; node: Record<string, any> }> = [];
  const seen = new Set<string>();
  let nodeId = linkNodeId(terminalLink);
  while (nodeId && nodeId !== baseNodeId && !seen.has(nodeId)) {
    seen.add(nodeId);
    const node = getNode(workflow, nodeId);
    if (!node || node.class_type !== 'LoraLoader' || !isRecord(node.inputs)) break;
    chain.push({ id: nodeId, node });
    nodeId = linkNodeId(node.inputs.model);
  }
  return chain.reverse();
}

function parseLoraChains(workflow: Record<string, any>, warnings: string[]): IdeogramWorkflowLora[] {
  const positiveChain = traceLoraModelChain(workflow, input(workflow, '98:157', 'model'), '98:23');
  const negativeChain = traceLoraModelChain(workflow, input(workflow, '98:155', 'model_negative'), '98:154');
  if (positiveChain.length === 0 && negativeChain.length === 0) return [];
  if (positiveChain.length !== negativeChain.length) {
    warnings.push('Imported LoRA chain has mismatched positive and negative LoRA counts; using matched entries where possible.');
  }

  const count = Math.max(positiveChain.length, negativeChain.length);
  const loras: IdeogramWorkflowLora[] = [];
  for (let index = 0; index < count; index += 1) {
    const positiveInputs = positiveChain[index]?.node.inputs || {};
    const negativeInputs = negativeChain[index]?.node.inputs || {};
    const positiveName = cleanText(positiveInputs.lora_name).trim();
    const negativeName = cleanText(negativeInputs.lora_name).trim();
    const loraName = positiveName || negativeName;
    if (!loraName) continue;
    if (positiveName && negativeName && positiveName !== negativeName) {
      warnings.push(`Imported LoRA pair ${index + 1} has mismatched names: ${positiveName} / ${negativeName}.`);
    }
    const positiveStrengthModel = finiteNumber(positiveInputs.strength_model, NaN);
    const negativeStrengthModel = finiteNumber(negativeInputs.strength_model, NaN);
    const positiveStrengthClip = finiteNumber(positiveInputs.strength_clip, NaN);
    const negativeStrengthClip = finiteNumber(negativeInputs.strength_clip, NaN);
    if (
      Number.isFinite(positiveStrengthModel) &&
      Number.isFinite(negativeStrengthModel) &&
      positiveStrengthModel !== negativeStrengthModel
    ) {
      warnings.push(`Imported LoRA pair ${index + 1} has mismatched model strengths; using the positive value.`);
    }
    if (
      Number.isFinite(positiveStrengthClip) &&
      Number.isFinite(negativeStrengthClip) &&
      positiveStrengthClip !== negativeStrengthClip
    ) {
      warnings.push(`Imported LoRA pair ${index + 1} has mismatched CLIP strengths; using the positive value.`);
    }
    loras.push({
      loraName,
      strengthModel: finiteNumber(positiveInputs.strength_model, finiteNumber(negativeInputs.strength_model, 1)),
      strengthClip: finiteNumber(positiveInputs.strength_clip, finiteNumber(negativeInputs.strength_clip, 1)),
    });
  }
  return loras;
}

export function classTypesFromWorkflow(workflow: Record<string, unknown>) {
  return Array.from(
    new Set(
      Object.values(workflow)
        .map(node => (isRecord(node) && typeof node.class_type === 'string' ? node.class_type : ''))
        .filter(Boolean),
    ),
  ).sort();
}

export function parseIdeogramComfyWorkflow(rawWorkflow: unknown): IdeogramImportResult {
  const warnings: string[] = [];
  const workflow = isRecord(rawWorkflow) ? deepClone(rawWorkflow) : buildIdeogramComfyWorkflow();
  if (!isRecord(rawWorkflow)) warnings.push('Imported workflow was not an object; loaded default Ideogram 4 FP8 workflow.');

  let next = cloneIdeogramWorkflowState();
  const promptText = input(workflow, '98:24', 'text');
  if (typeof promptText === 'string') {
    const parsed = parseIdeogramCaption(promptText);
    if (parsed.kind === 'ideogram') {
      next = ideogramDataToState(cloneIdeogramData(parsed.data), next);
    } else {
      warnings.push('Positive prompt was not recognized as Ideogram JSON; prompt fields were left at defaults.');
    }
  } else {
    warnings.push('Could not find the supported CLIPTextEncode prompt node.');
  }

  next.aspectRatio = cleanText(input(workflow, '37', 'aspect_ratio'), next.aspectRatio);
  next.megapixels = finiteNumber(input(workflow, '37', 'megapixels'), next.megapixels);
  next.seed = finiteInt(input(workflow, '98:18', 'noise_seed'), next.seed);
  next.samplerName = cleanText(input(workflow, '98:16', 'sampler_name'), next.samplerName);
  next.filenamePrefix = cleanText(input(workflow, '158', 'filename_prefix'), next.filenamePrefix);
  next.guiderCfg = finiteNumber(input(workflow, '98:155', 'cfg'), next.guiderCfg);
  next.overrideCfg = finiteNumber(input(workflow, '98:157', 'cfg'), next.overrideCfg);
  next.overrideStartPercent = finiteNumber(input(workflow, '98:157', 'start_percent'), next.overrideStartPercent);
  next.overrideEndPercent = finiteNumber(input(workflow, '98:157', 'end_percent'), next.overrideEndPercent);

  const qualityChoice = cleanText(input(workflow, '98:156', 'choice'), next.qualityPreset);
  next.qualityPreset = QUALITY_OPTIONS.includes(qualityChoice as IdeogramQualityPreset)
    ? (qualityChoice as IdeogramQualityPreset)
    : next.qualityPreset;
  const selectedPreset = IDEOGRAM4_QUALITY_PRESETS[next.qualityPreset] || IDEOGRAM4_QUALITY_PRESETS.Default;
  next.steps = positiveFiniteInt(input(workflow, '98:151', 'value'), selectedPreset.steps);

  const unetLoaders = findNodesByClass(workflow, 'UNETLoader');
  const unetNames = unetLoaders
    .map(({ node }) => (isRecord(node.inputs) ? node.inputs.unet_name : ''))
    .filter((value): value is string => typeof value === 'string' && value.length > 0);
  const unconditional = unetNames.find(name => name.toLowerCase().includes('unconditional')) || cleanText(input(workflow, '98:154', 'unet_name'));
  const diffusion =
    cleanText(input(workflow, '98:23', 'unet_name')) ||
    unetNames.find(name => name !== unconditional) ||
    next.models.diffusion;

  next.models = {
    diffusion,
    unconditional: unconditional || next.models.unconditional,
    clip: cleanText(input(workflow, '98:14', 'clip_name'), next.models.clip),
    vae: cleanText(input(workflow, '98:9', 'vae_name'), next.models.vae),
  };
  next.loras = parseLoraChains(workflow, warnings);

  const missingClasses = IDEOGRAM4_REQUIRED_NODE_CLASSES.filter(classType => !classTypesFromWorkflow(workflow).includes(classType));
  if (missingClasses.length > 0) {
    warnings.push(`Workflow is missing supported Ideogram nodes: ${missingClasses.join(', ')}`);
  }

  return { state: next, workflow, warnings };
}
