import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

function readSource(relativePath) {
  return fs.readFileSync(path.join(process.cwd(), relativePath), 'utf8');
}

function archBlock(source, name) {
  const start = source.indexOf(`name: '${name}'`);
  assert.notEqual(start, -1, `missing ${name} arch block`);
  const end = source.indexOf('\n  },', start);
  assert.notEqual(end, -1, `missing end of ${name} arch block`);
  return source.slice(start, end);
}

function assertKreaEditArch(block) {
  assert.match(block, /edit: true/);
  assert.match(block, /match_target_res: true/);
  assert.match(block, /kv_cache: true/);
  assert.match(block, /'config\.process\[0\]\.train\.unload_text_encoder': \[false, false\]/);
  assert.match(block, /'train\.unload_text_encoder'/);
  assert.match(block, /'datasets\.multi_control_paths'/);
  assert.match(block, /'sample\.multi_ctrl_imgs'/);
  assert.match(block, /'model\.qie\.match_target_res'/);
  assert.match(block, /'model\.model_kwargs\.kv_cache'/);
}

test('Krea2 edit UI presets expose edit controls and keep text encoder loaded', () => {
  const options = readSource('src/app/jobs/new/options.ts');

  const rawEdit = archBlock(options, 'krea2:o_edit');
  assertKreaEditArch(rawEdit);
  assert.match(rawEdit, /'config\.process\[0\]\.model\.name_or_path': \['krea\/Krea-2-Raw', defaultNameOrPath\]/);
  assert.doesNotMatch(rawEdit, /assistant_lora_path/);

  const turboEdit = archBlock(options, 'krea2:o_edit_turbo');
  assertKreaEditArch(turboEdit);
  assert.match(turboEdit, /'config\.process\[0\]\.model\.name_or_path': \['krea\/Krea-2-Turbo', defaultNameOrPath\]/);
  assert.match(turboEdit, /'model\.assistant_lora_path'/);
  assert.match(turboEdit, /krea2_turbo_training_adapter_v1\.safetensors/);
});
