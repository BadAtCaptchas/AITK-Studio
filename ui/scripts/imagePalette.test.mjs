import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  PALETTE_DELTA_E_THRESHOLDS,
  arePaletteColorsNearDuplicates,
  deltaE2000,
  extractPaletteFromRgba,
  findNearDuplicatePalettePairs,
  hexToLab,
  hexToRgb,
  paletteColorDistance,
} = require('../dist/src/utils/imagePalette.js');

function pixels(values) {
  return new Uint8ClampedArray(values.flat());
}

function lab(lightness, a, b) {
  return { lightness, a, b };
}

const SHARMA_REFERENCE_PAIRS = [
  [lab(50, 2.6772, -79.7751), lab(50, 0, -82.7485), 2.0425],
  [lab(50, 3.1571, -77.2803), lab(50, 0, -82.7485), 2.8615],
  [lab(50, 2.8361, -74.02), lab(50, 0, -82.7485), 3.4412],
  [lab(50, -1.3802, -84.2814), lab(50, 0, -82.7485), 1],
  [lab(50, -1.1848, -84.8006), lab(50, 0, -82.7485), 1],
  [lab(50, -0.9009, -85.5211), lab(50, 0, -82.7485), 1],
  [lab(50, 0, 0), lab(50, -1, 2), 2.3669],
  [lab(50, -1, 2), lab(50, 0, 0), 2.3669],
  [lab(50, 2.49, -0.001), lab(50, -2.49, 0.0009), 7.1792],
  [lab(50, 2.49, -0.001), lab(50, -2.49, 0.001), 7.1792],
  [lab(50, 2.49, -0.001), lab(50, -2.49, 0.0011), 7.2195],
  [lab(50, 2.49, -0.001), lab(50, -2.49, 0.0012), 7.2195],
  [lab(50, -0.001, 2.49), lab(50, 0.0009, -2.49), 4.8045],
  [lab(50, -0.001, 2.49), lab(50, 0.001, -2.49), 4.8045],
  [lab(50, -0.001, 2.49), lab(50, 0.0011, -2.49), 4.7461],
  [lab(50, 2.5, 0), lab(50, 0, -2.5), 4.3065],
  [lab(50, 2.5, 0), lab(73, 25, -18), 27.1492],
  [lab(50, 2.5, 0), lab(61, -5, 29), 22.8977],
  [lab(50, 2.5, 0), lab(56, -27, -3), 31.903],
  [lab(50, 2.5, 0), lab(58, 24, 15), 19.4535],
  [lab(50, 2.5, 0), lab(50, 3.1736, 0.5854), 1],
  [lab(50, 2.5, 0), lab(50, 3.2972, 0), 1],
  [lab(50, 2.5, 0), lab(50, 1.8634, 0.5757), 1],
  [lab(50, 2.5, 0), lab(50, 3.2592, 0.335), 1],
  [lab(60.2574, -34.0099, 36.2677), lab(60.4626, -34.1751, 39.4387), 1.2644],
  [lab(63.0109, -31.0961, -5.8663), lab(62.8187, -29.7946, -4.0864), 1.263],
  [lab(61.2901, 3.7196, -5.3901), lab(61.4292, 2.248, -4.962), 1.8731],
  [lab(35.0831, -44.1164, 3.7933), lab(35.0232, -40.0716, 1.5901), 1.8645],
  [lab(22.7233, 20.0904, -46.694), lab(23.0331, 14.973, -42.5619), 2.0373],
  [lab(36.4612, 47.858, 18.3852), lab(36.2715, 50.5065, 21.2231), 1.4146],
  [lab(90.8027, -2.0831, 1.441), lab(91.1528, -1.6435, 0.0447), 1.4441],
  [lab(90.9257, -0.5406, -0.9208), lab(88.6381, -0.8985, -0.7239), 1.5381],
  [lab(6.7747, -0.2908, -2.4247), lab(5.8714, -0.0985, -2.2286), 0.6377],
  [lab(2.0776, 0.0795, -1.135), lab(0.9033, -0.0636, -0.5514), 0.9082],
];

test('palette extraction is deterministic and ordered by sampled coverage', () => {
  const rgba = pixels([
    [232, 226, 214, 255],
    [232, 226, 214, 255],
    [232, 226, 214, 255],
    [63, 49, 38, 255],
    [63, 49, 38, 255],
    [212, 175, 55, 255],
  ]);
  const result = extractPaletteFromRgba(rgba, 6, 1, { maxColors: 3, maxSamples: 256 });

  assert.equal(result.sampledPixels, 6);
  assert.deepEqual(
    result.swatches.map(swatch => swatch.color),
    ['#E8E2D6', '#3F3126', '#D4AF37'],
  );
  assert.ok(result.swatches[0].coverage > result.swatches[1].coverage);
  assert.equal(Math.round(result.coverage), 100);
});

test('palette extraction ignores transparent pixels and caps the result', () => {
  const rgba = pixels([
    [255, 0, 255, 0],
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
  ]);
  const result = extractPaletteFromRgba(rgba, 4, 1, { maxColors: 2, maxSamples: 256 });

  assert.equal(result.sampledPixels, 3);
  assert.equal(result.swatches.length, 2);
  assert.equal(
    result.swatches.some(swatch => swatch.color === '#FF00FF'),
    false,
  );
});

test('stratified sampling retains both colors in repeating pixel patterns', () => {
  const width = 32;
  const height = 32;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const firstColor = (x + y) % 2 === 0;
      rgba[offset] = firstColor ? 255 : 0;
      rgba[offset + 1] = 0;
      rgba[offset + 2] = firstColor ? 0 : 255;
      rgba[offset + 3] = 255;
    }
  }

  const result = extractPaletteFromRgba(rgba, width, height, { maxColors: 2, maxSamples: 256 });
  assert.deepEqual(new Set(result.swatches.map(swatch => swatch.color)), new Set(['#FF0000', '#0000FF']));
  assert.deepEqual(
    result.swatches.map(swatch => Math.round(swatch.coverage)).sort((left, right) => left - right),
    [50, 50],
  );
});

test('stratified sampling honors the sample budget for extreme aspect ratios', () => {
  const width = 10_000;
  const rgba = new Uint8ClampedArray(width * 4);
  for (let pixelIndex = 0; pixelIndex < width; pixelIndex += 1) rgba[pixelIndex * 4 + 3] = 255;

  const result = extractPaletteFromRgba(rgba, width, 1, { maxColors: 1, maxSamples: 256 });
  assert.equal(result.sampledPixels, 256);
});

test('CIEDE2000 matches all published Sharma reference pairs', () => {
  for (const [reference, sample, expected] of SHARMA_REFERENCE_PAIRS) {
    const actual = deltaE2000(reference, sample);
    assert.ok(Math.abs(actual - expected) <= 0.0001, `Expected ${expected.toFixed(4)}, received ${actual.toFixed(6)}`);
  }
});

test('sRGB conversion and perceptual distance handle identity, symmetry, and invalid input', () => {
  assert.deepEqual(hexToRgb('#E8E2D6'), { red: 232, green: 226, blue: 214 });
  assert.equal(hexToRgb('#FFF'), null);

  const black = hexToLab('#000000');
  const white = hexToLab('#FFFFFF');
  assert.ok(black && Math.abs(black.lightness) < 0.0001 && Math.abs(black.a) < 0.0001 && Math.abs(black.b) < 0.0001);
  assert.ok(
    white && Math.abs(white.lightness - 100) < 0.0001 && Math.abs(white.a) < 0.0001 && Math.abs(white.b) < 0.0001,
  );

  assert.equal(paletteColorDistance('#000000', '#000000'), 0);
  assert.ok(Math.abs(paletteColorDistance('#E8E2D6', '#C8BDAF') - paletteColorDistance('#C8BDAF', '#E8E2D6')) < 1e-12);
  assert.equal(paletteColorDistance('bad', '#FFFFFF'), Number.POSITIVE_INFINITY);
});

test('near-duplicate classification rejects the reported gray and ochre pair', () => {
  const closeGrayDistance = paletteColorDistance('#6C6F68', '#6D7069');
  const reportedDistance = paletteColorDistance('#6C6F68', '#B87945');

  assert.ok(closeGrayDistance < PALETTE_DELTA_E_THRESHOLDS.nearDuplicate);
  assert.equal(arePaletteColorsNearDuplicates('#6C6F68', '#6D7069'), true);
  assert.ok(reportedDistance > 20);
  assert.equal(arePaletteColorsNearDuplicates('#6C6F68', '#B87945'), false);
});

test('duplicate analysis ranks closest pairs and never reuses a color', () => {
  const pairs = findNearDuplicatePalettePairs(['#6C6F68', '#BA7A47', '#6D7069', '#B87945', 'invalid']);

  assert.equal(pairs.length, 2);
  assert.deepEqual(pairs[0].colors, ['#6C6F68', '#6D7069']);
  assert.deepEqual(pairs[1].colors, ['#BA7A47', '#B87945']);
  assert.ok(pairs[0].distance < pairs[1].distance);
  assert.equal(new Set(pairs.flatMap(pair => pair.colors)).size, 4);
});

test('extraction keeps the reported colors separate and does not borrow coverage', () => {
  const rgba = pixels([
    [108, 111, 104, 255],
    [108, 111, 104, 255],
    [108, 111, 104, 255],
    [184, 121, 69, 255],
    [184, 121, 69, 255],
    [184, 121, 69, 255],
  ]);
  const fullPalette = extractPaletteFromRgba(rgba, 6, 1, { maxColors: 2, maxSamples: 256 });
  const singleSwatch = extractPaletteFromRgba(rgba, 6, 1, { maxColors: 1, maxSamples: 256 });

  assert.deepEqual(new Set(fullPalette.swatches.map(swatch => swatch.color)), new Set(['#6C6F68', '#B87945']));
  assert.equal(Math.round(fullPalette.coverage), 100);
  assert.equal(Math.round(singleSwatch.coverage), 50);
});
