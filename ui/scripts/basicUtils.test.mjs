import assert from 'node:assert/strict';
import test from 'node:test';
import { getFileExtension, isAudio, isImage, isVideo } from '../dist/src/utils/basic.js';

test('getFileExtension handles multi-character image extensions', () => {
  assert.equal(getFileExtension('C:\\datasets\\sample.WEBP'), '.webp');
  assert.equal(getFileExtension('/datasets/sample.JXL'), '.jxl');
  assert.equal(getFileExtension('/datasets/sample.jpeg?cache=1'), '.jpeg');
  assert.equal(getFileExtension('/datasets/no_extension'), '');
});

test('media type helpers classify WebP and long extensions', () => {
  assert.equal(isImage('/datasets/sample.webp'), true);
  assert.equal(isImage('/datasets/sample.jxl'), true);
  assert.equal(isImage('/datasets/sample.jpeg'), true);
  assert.equal(isVideo('/datasets/clip.m4v'), true);
  assert.equal(isAudio('/datasets/audio.flac'), true);
});
