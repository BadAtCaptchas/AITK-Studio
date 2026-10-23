import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource } from './trainingMigrationFixture.mjs';

const { parseTrainingConfigImport } = loadTrainingSource('utils/trainingConfigImport.ts');

test('JSON and JSONC imports preserve URLs, literal comment markers and escaped string contents', () => {
  const expected = {
    model: 'https://models.invalid/base?revision=main',
    prompt: 'a sign saying /* KEEP THIS */ // literally, "quoted", and C:\\models\\one',
    escaped: '\\" // still inside the string',
  };
  for (const filename of ['training.json', 'training.jsonc', 'TRAINING.JSONC']) {
    assert.deepEqual(parseTrainingConfigImport(JSON.stringify(expected), filename), expected);
  }
});

test('actual JSON comments are ignored while whitespace and tokens stay separate', () => {
  const text = '\uFEFF/* header */\n{\n// model comment\n"url": "https://models.invalid/one", /* inline */\n"value": 12, "enabled": true\n}// ending';
  assert.deepEqual(parseTrainingConfigImport(text, 'training.jsonc'), {
    url: 'https://models.invalid/one', value: 12, enabled: true,
  });
  assert.throws(() => parseTrainingConfigImport('{"value": 1/* separator */2}', 'training.jsonc'));
  assert.throws(() => parseTrainingConfigImport('{"value": 1} /* unterminated', 'training.jsonc'), /Unterminated/);
  assert.throws(() => parseTrainingConfigImport('{"value": "unterminated}', 'training.json'));
});

test('YAML imports retain multiline prompts and comment-like text', () => {
  assert.deepEqual(parseTrainingConfigImport('prompt: |\n  https://models.invalid/one\n  /* literal */\n', 'training.yaml'), {
    prompt: 'https://models.invalid/one\n/* literal */\n',
  });
});

test('cyclic YAML imports cannot enter a draft that must be serialized for saving', () => {
  assert.throws(() => parseTrainingConfigImport('model_kwargs: &recursive\n  nested: *recursive\n', 'training.yaml'), /circular/i);
  assert.deepEqual(parseTrainingConfigImport('first: &shared\n  value: 1\nsecond: *shared\n', 'training.yaml'), {
    first: { value: 1 }, second: { value: 1 },
  });
});
