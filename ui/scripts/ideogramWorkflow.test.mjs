import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

const require = createRequire(import.meta.url);

const {
  buildIdeogramComfyWorkflow,
  classTypesFromWorkflow,
  closestIdeogramAspectRatio,
  DEFAULT_IDEOGRAM_WORKFLOW_STATE,
  parseIdeogramAspectRatio,
  parseIdeogramComfyWorkflow,
  requiredIdeogramModels,
} = require('../dist/src/utils/ideogramWorkflow.js');

const {
  ExternalComfyError,
  copyToolkitLoraToExternalComfy,
  getComfyViewImage,
  imageRefsFromHistoryEntry,
  listExternalComfyLoras,
  loraNamesFromObjectInfo,
  normalizeComfyHistoryEntry,
  queueComfyPrompt,
  runIdeogramComfyPreflight,
  workflowFromHistoryEntry,
} = require('../dist/src/server/externalComfy.js');

function responseJson(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function responseBytes(bytes, status = 200) {
  return new Response(Uint8Array.from(bytes), {
    status,
    headers: { 'Content-Type': 'image/png' },
  });
}

function objectInfoForWorkflow(workflow, overrides = {}) {
  const objectInfo = Object.fromEntries(
    classTypesFromWorkflow(workflow).map(classType => [
      classType,
      {
        input: {
          required: {},
        },
      },
    ]),
  );

  objectInfo.UNETLoader = {
    input: {
      required: {
        unet_name: [[
          DEFAULT_IDEOGRAM_WORKFLOW_STATE.models.diffusion,
          DEFAULT_IDEOGRAM_WORKFLOW_STATE.models.unconditional,
        ]],
      },
    },
  };
  objectInfo.CLIPLoader = {
    input: {
      required: {
        clip_name: [[DEFAULT_IDEOGRAM_WORKFLOW_STATE.models.clip]],
      },
    },
  };
  objectInfo.VAELoader = {
    input: {
      required: {
        vae_name: [[DEFAULT_IDEOGRAM_WORKFLOW_STATE.models.vae]],
      },
    },
  };
  const loraNames = Array.from(
    new Set(
      Object.values(workflow)
        .map(node => node?.inputs?.lora_name)
        .filter(value => typeof value === 'string' && value.length > 0),
    ),
  );
  if (loraNames.length > 0) {
    objectInfo.LoraLoader = {
      input: {
        required: {
          lora_name: [loraNames],
        },
      },
    };
  }

  return { ...objectInfo, ...overrides };
}

test('buildIdeogramComfyWorkflow maps builder state into the FP8 API workflow', () => {
  const state = {
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    seed: 12345,
    qualityPreset: 'Quality',
    steps: 36,
    guiderCfg: 6.5,
    filenamePrefix: 'unit_test_ideogram',
  };

  const workflow = buildIdeogramComfyWorkflow(state);
  assert.equal(workflow['37'].inputs.aspect_ratio, state.aspectRatio);
  assert.equal(workflow['98:18'].inputs.noise_seed, 12345);
  assert.equal(workflow['98:23'].inputs.unet_name, state.models.diffusion);
  assert.equal(workflow['98:154'].inputs.unet_name, state.models.unconditional);
  assert.equal(workflow['98:14'].inputs.clip_name, state.models.clip);
  assert.equal(workflow['98:9'].inputs.vae_name, state.models.vae);
  assert.equal(workflow['98:156'].inputs.choice, 'Quality');
  assert.equal(workflow['98:151'].inputs.value, 36);
  assert.equal(workflow['98:155'].inputs.cfg, 6.5);
  assert.equal(workflow['158'].inputs.filename_prefix, 'unit_test_ideogram');
  assert.equal(classTypesFromWorkflow(workflow).includes('LoraLoader'), false);
  assert.deepEqual(workflow['98:157'].inputs.model, ['98:23', 0]);
  assert.deepEqual(workflow['98:155'].inputs.model_negative, ['98:154', 0]);
  assert.deepEqual(workflow['98:24'].inputs.clip, ['98:14', 0]);

  const prompt = JSON.parse(workflow['98:24'].inputs.text);
  assert.equal(prompt.high_level_description, state.highLevelDescription);
  assert.equal(prompt.compositional_deconstruction.elements.length, state.elements.length);
});

test('buildIdeogramComfyWorkflow exports matched positive and negative LoRA pairs', () => {
  const workflow = buildIdeogramComfyWorkflow({
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    loras: [
      { loraName: 'chronoedit_distill_lora.safetensors', strengthModel: 0.85, strengthClip: 0.75 },
    ],
  });

  assert.equal(workflow['98:177'].class_type, 'LoraLoader');
  assert.equal(workflow['98:178'].class_type, 'LoraLoader');
  assert.equal(workflow['98:177'].inputs.lora_name, 'chronoedit_distill_lora.safetensors');
  assert.equal(workflow['98:178'].inputs.lora_name, 'chronoedit_distill_lora.safetensors');
  assert.equal(workflow['98:177'].inputs.strength_model, 0.85);
  assert.equal(workflow['98:178'].inputs.strength_model, 0.85);
  assert.equal(workflow['98:177'].inputs.strength_clip, 0.75);
  assert.equal(workflow['98:178'].inputs.strength_clip, 0.75);
  assert.deepEqual(workflow['98:177'].inputs.model, ['98:23', 0]);
  assert.deepEqual(workflow['98:178'].inputs.model, ['98:154', 0]);
  assert.deepEqual(workflow['98:177'].inputs.clip, ['98:14', 0]);
  assert.deepEqual(workflow['98:178'].inputs.clip, ['98:14', 0]);
  assert.deepEqual(workflow['98:157'].inputs.model, ['98:177', 0]);
  assert.deepEqual(workflow['98:155'].inputs.model_negative, ['98:178', 0]);
  assert.deepEqual(workflow['98:24'].inputs.clip, ['98:178', 1]);
});

test('buildIdeogramComfyWorkflow chains multiple LoRAs deterministically', () => {
  const workflow = buildIdeogramComfyWorkflow({
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    loras: [
      { loraName: 'first.safetensors', strengthModel: 1, strengthClip: 1 },
      { loraName: 'second.safetensors', strengthModel: 0.6, strengthClip: 0.5 },
    ],
  });

  assert.equal(workflow['98:177'].inputs.lora_name, 'first.safetensors');
  assert.equal(workflow['98:178'].inputs.lora_name, 'first.safetensors');
  assert.equal(workflow['98:179'].inputs.lora_name, 'second.safetensors');
  assert.equal(workflow['98:180'].inputs.lora_name, 'second.safetensors');
  assert.deepEqual(workflow['98:179'].inputs.model, ['98:177', 0]);
  assert.deepEqual(workflow['98:180'].inputs.model, ['98:178', 0]);
  assert.deepEqual(workflow['98:179'].inputs.clip, ['98:178', 1]);
  assert.deepEqual(workflow['98:180'].inputs.clip, ['98:178', 1]);
  assert.deepEqual(workflow['98:157'].inputs.model, ['98:179', 0]);
  assert.deepEqual(workflow['98:155'].inputs.model_negative, ['98:180', 0]);
  assert.deepEqual(workflow['98:24'].inputs.clip, ['98:180', 1]);
});

test('parseIdeogramComfyWorkflow imports the supported workflow shape and warns on unsupported JSON', () => {
  const workflow = buildIdeogramComfyWorkflow({
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    seed: 9876,
    qualityPreset: 'Turbo',
    steps: 17,
    guiderCfg: 3.25,
    aspectRatio: '16:9 (Widescreen)',
  });
  const imported = parseIdeogramComfyWorkflow(workflow);

  assert.equal(imported.state.seed, 9876);
  assert.equal(imported.state.qualityPreset, 'Turbo');
  assert.equal(imported.state.steps, 17);
  assert.equal(imported.state.guiderCfg, 3.25);
  assert.equal(imported.state.aspectRatio, '16:9 (Widescreen)');
  assert.equal(imported.state.models.diffusion, DEFAULT_IDEOGRAM_WORKFLOW_STATE.models.diffusion);
  assert.equal(imported.state.elements[0].type, DEFAULT_IDEOGRAM_WORKFLOW_STATE.elements[0].type);
  assert.deepEqual(imported.warnings, []);

  const unsupported = parseIdeogramComfyWorkflow({ '1': { class_type: 'SaveImage', inputs: {} } });
  assert.equal(unsupported.state.seed, DEFAULT_IDEOGRAM_WORKFLOW_STATE.seed);
  assert.ok(unsupported.warnings.some(message => message.includes('CLIPTextEncode')));
  assert.ok(unsupported.warnings.some(message => message.includes('missing supported Ideogram nodes')));
});

test('parseIdeogramComfyWorkflow roundtrips LoRA chains and warns on mismatched pairs', () => {
  const workflow = buildIdeogramComfyWorkflow({
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    aspectRatio: '1:1 (Square)',
    steps: 24,
    guiderCfg: 4.75,
    loras: [
      { loraName: 'chronoedit_distill_lora.safetensors', strengthModel: 0.9, strengthClip: 0.8 },
      { loraName: 'poster_detail.safetensors', strengthModel: 0.55, strengthClip: 0.45 },
    ],
  });
  const imported = parseIdeogramComfyWorkflow(workflow);

  assert.equal(imported.state.aspectRatio, '1:1 (Square)');
  assert.equal(imported.state.steps, 24);
  assert.equal(imported.state.guiderCfg, 4.75);
  assert.deepEqual(imported.state.loras, [
    { loraName: 'chronoedit_distill_lora.safetensors', strengthModel: 0.9, strengthClip: 0.8 },
    { loraName: 'poster_detail.safetensors', strengthModel: 0.55, strengthClip: 0.45 },
  ]);
  assert.deepEqual(imported.warnings, []);

  workflow['98:178'].inputs.lora_name = 'different_negative.safetensors';
  workflow['98:178'].inputs.strength_model = 0.2;
  const mismatched = parseIdeogramComfyWorkflow(workflow);
  assert.equal(mismatched.state.loras[0].loraName, 'chronoedit_distill_lora.safetensors');
  assert.equal(mismatched.state.loras[0].strengthModel, 0.9);
  assert.ok(mismatched.warnings.some(message => message.includes('mismatched names')));
  assert.ok(mismatched.warnings.some(message => message.includes('mismatched model strengths')));
});

test('Ideogram aspect helpers parse supported aspect ratios and match imported images', () => {
  assert.equal(parseIdeogramAspectRatio('1:1 (Square)').css, '1 / 1');
  assert.equal(parseIdeogramAspectRatio('2:3 (Portrait Photo)').css, '2 / 3');
  assert.equal(parseIdeogramAspectRatio('3:2 (Landscape Photo)').css, '3 / 2');
  assert.equal(parseIdeogramAspectRatio('4:5 (Portrait)').css, '4 / 5');
  assert.equal(parseIdeogramAspectRatio('9:16 (Vertical)').css, '9 / 16');
  assert.equal(parseIdeogramAspectRatio('16:9 (Widescreen)').css, '16 / 9');
  assert.equal(closestIdeogramAspectRatio(1920, 1080), '16:9 (Widescreen)');
  assert.equal(closestIdeogramAspectRatio(1080, 1920), '9:16 (Vertical)');
});

test('runIdeogramComfyPreflight classifies nodes and model files from object_info', async () => {
  const workflow = buildIdeogramComfyWorkflow();
  const fetchImpl = async url => {
    const href = String(url);
    if (href.endsWith('/system_stats')) return responseJson({ system: 'ok' });
    if (href.endsWith('/object_info')) return responseJson(objectInfoForWorkflow(workflow));
    throw new Error(`unexpected ${href}`);
  };

  const result = await runIdeogramComfyPreflight({
    serverUrl: 'http://127.0.0.1:8188',
    workflow,
    models: requiredIdeogramModels(DEFAULT_IDEOGRAM_WORKFLOW_STATE),
    fetchImpl,
  });

  assert.equal(result.ok, true);
  assert.equal(result.connected, true);
  assert.equal(result.models.every(item => item.status === 'found'), true);
  assert.equal(result.nodes.every(item => item.status === 'found'), true);

  const missingFetch = async url => {
    const href = String(url);
    if (href.endsWith('/system_stats')) return responseJson({ system: 'ok' });
    if (href.endsWith('/object_info')) {
      return responseJson(
        objectInfoForWorkflow(workflow, {
          UNETLoader: { input: { required: { unet_name: [['only_other_model.safetensors']] } } },
        }),
      );
    }
    throw new Error(`unexpected ${href}`);
  };
  const missing = await runIdeogramComfyPreflight({
    serverUrl: 'http://127.0.0.1:8188',
    workflow,
    models: requiredIdeogramModels(DEFAULT_IDEOGRAM_WORKFLOW_STATE),
    fetchImpl: missingFetch,
  });
  assert.equal(missing.ok, false);
  assert.equal(missing.models.filter(item => item.status === 'missing').length, 2);
});

test('external Comfy LoRA discovery reads object_info and falls back to /models/loras', async () => {
  const objectInfo = {
    LoraLoader: {
      input: {
        required: {
          lora_name: [['zeta.safetensors', 'alpha.safetensors', 'alpha.safetensors']],
        },
      },
    },
  };
  assert.deepEqual(loraNamesFromObjectInfo(objectInfo), ['zeta.safetensors', 'alpha.safetensors', 'alpha.safetensors']);

  const discovered = await listExternalComfyLoras('http://127.0.0.1:8188', async url => {
    const href = String(url);
    if (href.endsWith('/object_info')) return responseJson(objectInfo);
    throw new Error(`unexpected ${href}`);
  });
  assert.deepEqual(discovered, {
    source: 'object_info',
    loras: ['alpha.safetensors', 'zeta.safetensors'],
  });

  const fallback = await listExternalComfyLoras('http://127.0.0.1:8188', async url => {
    const href = String(url);
    if (href.endsWith('/object_info')) return responseJson({ LoraLoader: { input: { required: {} } } });
    if (href.endsWith('/models/loras')) return responseJson(['beta.safetensors', 'alpha.safetensors']);
    throw new Error(`unexpected ${href}`);
  });
  assert.deepEqual(fallback, {
    source: 'models',
    loras: ['alpha.safetensors', 'beta.safetensors'],
  });
});

test('runIdeogramComfyPreflight reports missing selected LoRAs', async () => {
  const state = {
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    loras: [{ loraName: 'missing_lora.safetensors', strengthModel: 1, strengthClip: 1 }],
  };
  const workflow = buildIdeogramComfyWorkflow(state);
  const fetchImpl = async url => {
    const href = String(url);
    if (href.endsWith('/system_stats')) return responseJson({ system: 'ok' });
    if (href.endsWith('/object_info')) {
      return responseJson(
        objectInfoForWorkflow(workflow, {
          LoraLoader: { input: { required: { lora_name: [['installed_lora.safetensors']] } } },
        }),
      );
    }
    throw new Error(`unexpected ${href}`);
  };

  const result = await runIdeogramComfyPreflight({
    serverUrl: 'http://127.0.0.1:8188',
    workflow,
    models: requiredIdeogramModels(state),
    fetchImpl,
  });
  assert.equal(result.ok, false);
  assert.ok(result.models.some(item => item.label.includes('missing_lora.safetensors') && item.status === 'missing'));
});

test('external Comfy helpers queue prompts, parse history entries, and fetch view bytes', async () => {
  const workflow = buildIdeogramComfyWorkflow({
    ...DEFAULT_IDEOGRAM_WORKFLOW_STATE,
    aspectRatio: '3:2 (Landscape Photo)',
    steps: 28,
    guiderCfg: 5.5,
  });
  const historyEntry = {
    prompt: [12, 'prompt-1', workflow],
    outputs: {
      '158': {
        images: [{ filename: 'Ideogram_00001_.png', subfolder: 'ideogram_fp8', type: 'output' }],
      },
    },
    status: { status_str: 'success' },
  };
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    const href = String(url);
    calls.push({ href, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null });
    if (href.endsWith('/prompt')) return responseJson({ prompt_id: 'prompt-1', number: 3 });
    if (href.endsWith('/history/prompt-1')) return responseJson({ 'prompt-1': historyEntry });
    if (href.includes('/view?')) return responseBytes([1, 2, 3, 4]);
    throw new Error(`unexpected ${href}`);
  };

  const queued = await queueComfyPrompt({
    serverUrl: 'http://127.0.0.1:8188',
    workflow,
    clientId: 'client-1',
    fetchImpl,
  });
  assert.deepEqual(queued, { promptId: 'prompt-1', queueNumber: 3, clientId: 'client-1' });
  assert.equal(calls[0].body.client_id, 'client-1');

  const history = await fetchImpl(new URL('http://127.0.0.1:8188/history/prompt-1')).then(res => res.json());
  const entry = normalizeComfyHistoryEntry(history, 'prompt-1');
  assert.deepEqual(workflowFromHistoryEntry(entry), workflow);
  const importedFromHistory = parseIdeogramComfyWorkflow(workflowFromHistoryEntry(entry));
  assert.equal(importedFromHistory.state.aspectRatio, '3:2 (Landscape Photo)');
  assert.equal(importedFromHistory.state.steps, 28);
  assert.equal(importedFromHistory.state.guiderCfg, 5.5);
  assert.deepEqual(imageRefsFromHistoryEntry(entry), [
    { filename: 'Ideogram_00001_.png', subfolder: 'ideogram_fp8', type: 'output' },
  ]);

  const image = await getComfyViewImage({
    serverUrl: 'http://127.0.0.1:8188',
    filename: 'Ideogram_00001_.png',
    subfolder: 'ideogram_fp8',
    type: 'output',
    fetchImpl,
  });
  assert.equal(image.byteLength, 4);
});

test('copyToolkitLoraToExternalComfy only copies known Toolkit LoRAs and refuses overwrites', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-lora-copy-'));
  try {
    const sourceDir = path.join(root, 'toolkit');
    const destinationDir = path.join(root, 'external-loras');
    await fs.mkdir(sourceDir, { recursive: true });
    await fs.mkdir(destinationDir, { recursive: true });
    const sourcePath = path.join(sourceDir, 'studio_lora.safetensors');
    const unknownPath = path.join(sourceDir, 'unknown_lora.safetensors');
    await fs.writeFile(sourcePath, 'known lora');
    await fs.writeFile(unknownPath, 'unknown lora');
    const knownLoras = [
      {
        id: 'uploaded:studio_lora.safetensors',
        label: 'Uploaded / studio_lora.safetensors',
        path: sourcePath,
        filename: 'studio_lora.safetensors',
        source: 'uploaded',
        sizeBytes: 10,
        updatedAt: new Date().toISOString(),
        triggerWords: [],
      },
    ];

    const copied = await copyToolkitLoraToExternalComfy({
      toolkitPath: sourcePath,
      loraDir: destinationDir,
      knownLoras,
    });
    assert.equal(copied.filename, 'studio_lora.safetensors');
    assert.equal(await fs.readFile(path.join(destinationDir, 'studio_lora.safetensors'), 'utf8'), 'known lora');

    await assert.rejects(
      () => copyToolkitLoraToExternalComfy({ toolkitPath: sourcePath, loraDir: destinationDir, knownLoras }),
      error => error instanceof ExternalComfyError && error.status === 409,
    );
    await assert.rejects(
      () => copyToolkitLoraToExternalComfy({ toolkitPath: unknownPath, loraDir: destinationDir, knownLoras }),
      error => error instanceof ExternalComfyError && error.status === 403,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('external Comfy helpers surface HTTP and malformed prompt errors', async () => {
  await assert.rejects(
    () =>
      queueComfyPrompt({
        serverUrl: 'http://127.0.0.1:8188',
        workflow: buildIdeogramComfyWorkflow(),
        clientId: 'client-1',
        fetchImpl: async () => responseJson({ number: 1 }),
      }),
    /did not return a prompt_id/,
  );

  await assert.rejects(
    () =>
      getComfyViewImage({
        serverUrl: 'http://127.0.0.1:8188',
        filename: 'missing.png',
        fetchImpl: async () => responseJson({ error: 'missing' }, 500),
      }),
    error => error instanceof ExternalComfyError && /ComfyUI request failed/.test(error.message),
  );
});
