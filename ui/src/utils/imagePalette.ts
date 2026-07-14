export type ImagePaletteSwatch = {
  color: string;
  sampleCount: number;
  coverage: number;
};

export type ImagePaletteExtraction = {
  swatches: ImagePaletteSwatch[];
  sampledPixels: number;
  representedPixels: number;
  coverage: number;
  confidence: number;
};

export type LabColor = {
  lightness: number;
  a: number;
  b: number;
};

export type PaletteSimilarityPair = {
  colors: readonly [string, string];
  /** Perceptual difference using CIEDE2000. Lower values are more alike. */
  distance: number;
};

type PaletteBucket = {
  key: number;
  count: number;
  red: number;
  green: number;
  blue: number;
};

type PaletteCluster = {
  bucket: PaletteBucket;
  lab: LabColor;
};

type Rgb = { red: number; green: number; blue: number };
type Vector3 = readonly [number, number, number];
type Matrix3 = readonly [Vector3, Vector3, Vector3];

/**
 * Separate policies are deliberate: final swatches must be almost
 * indistinguishable before suggesting a merge, while sampled pixels need a
 * wider radius to absorb lighting, compression, and anti-aliasing noise.
 */
export const PALETTE_DELTA_E_THRESHOLDS = {
  indistinguishable: 1,
  nearDuplicate: 2,
  swatchMatch: 12,
  extractionCluster: 5,
  representedCoverage: 15,
} as const;

const D50_WHITE: Vector3 = [0.3457 / 0.3585, 1, (1 - 0.3457 - 0.3585) / 0.3585];

// Exact sRGB-to-XYZ coefficients from CSS Color 4.
const LINEAR_SRGB_TO_XYZ_D65: Matrix3 = [
  [506752 / 1228815, 87881 / 245763, 12673 / 70218],
  [87098 / 409605, 175762 / 245763, 12673 / 175545],
  [7918 / 409605, 87881 / 737289, 1001167 / 1053270],
];

// Bradford chromatic adaptation from D65 to D50.
const XYZ_D65_TO_D50: Matrix3 = [
  [1.0479297925449969, 0.022946870601609652, -0.05019226628920524],
  [0.02962780877005599, 0.9904344267538799, -0.017073799063418826],
  [-0.009243040646204504, 0.015055191490298152, 0.7518742814281371],
];

function channelHex(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)))
    .toString(16)
    .padStart(2, '0')
    .toUpperCase();
}

function rgbToHex(color: Rgb) {
  return `#${channelHex(color.red)}${channelHex(color.green)}${channelHex(color.blue)}`;
}

export function hexToRgb(value: string): Rgb | null {
  const match = /^#([0-9A-Fa-f]{6})$/.exec(value.trim());
  if (!match) return null;
  return {
    red: Number.parseInt(match[1].slice(0, 2), 16),
    green: Number.parseInt(match[1].slice(2, 4), 16),
    blue: Number.parseInt(match[1].slice(4, 6), 16),
  };
}

function multiplyMatrixVector(matrix: Matrix3, vector: Vector3): Vector3 {
  const multiplyRow = (row: Vector3) => row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2];
  return [multiplyRow(matrix[0]), multiplyRow(matrix[1]), multiplyRow(matrix[2])];
}

function linearizeSrgbChannel(value: number) {
  const channel = Math.max(0, Math.min(255, value)) / 255;
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

/** Convert an sRGB color to D50-adapted CIELAB, following CSS Color 4. */
export function rgbToLab(color: Rgb): LabColor {
  const linearRgb: Vector3 = [
    linearizeSrgbChannel(color.red),
    linearizeSrgbChannel(color.green),
    linearizeSrgbChannel(color.blue),
  ];
  const xyzD65 = multiplyMatrixVector(LINEAR_SRGB_TO_XYZ_D65, linearRgb);
  const xyzD50 = multiplyMatrixVector(XYZ_D65_TO_D50, xyzD65);
  const epsilon = 216 / 24389;
  const kappa = 24389 / 27;
  const relativeXyz: Vector3 = [xyzD50[0] / D50_WHITE[0], xyzD50[1] / D50_WHITE[1], xyzD50[2] / D50_WHITE[2]];
  const transform = (value: number) => (value > epsilon ? Math.cbrt(value) : (kappa * value + 16) / 116);
  const transformed: Vector3 = [transform(relativeXyz[0]), transform(relativeXyz[1]), transform(relativeXyz[2])];

  return {
    lightness: 116 * transformed[1] - 16,
    a: 500 * (transformed[0] - transformed[1]),
    b: 200 * (transformed[1] - transformed[2]),
  };
}

export function hexToLab(value: string): LabColor | null {
  const color = hexToRgb(value);
  return color ? rgbToLab(color) : null;
}

/**
 * CIEDE2000 with the standard kL, kC, and kH weighting factors of 1.
 * The hue wrapping follows the W3C implementation validated against Sharma's
 * published reference pairs.
 */
export function deltaE2000(reference: LabColor, sample: LabColor) {
  const components = [reference.lightness, reference.a, reference.b, sample.lightness, sample.a, sample.b];
  if (!components.every(Number.isFinite)) return Number.POSITIVE_INFINITY;

  const chroma1 = Math.hypot(reference.a, reference.b);
  const chroma2 = Math.hypot(sample.a, sample.b);
  const meanChroma = (chroma1 + chroma2) / 2;
  const meanChroma7 = meanChroma ** 7;
  const twentyFive7 = 25 ** 7;
  const asymmetry = 0.5 * (1 - Math.sqrt(meanChroma7 / (meanChroma7 + twentyFive7)));
  const adjustedA1 = (1 + asymmetry) * reference.a;
  const adjustedA2 = (1 + asymmetry) * sample.a;
  const adjustedChroma1 = Math.hypot(adjustedA1, reference.b);
  const adjustedChroma2 = Math.hypot(adjustedA2, sample.b);
  const radiansToDegrees = 180 / Math.PI;
  const degreesToRadians = Math.PI / 180;

  const hueDegrees = (a: number, b: number) => {
    if (a === 0 && b === 0) return 0;
    const angle = Math.atan2(b, a) * radiansToDegrees;
    return angle < 0 ? angle + 360 : angle;
  };

  const hue1 = hueDegrees(adjustedA1, reference.b);
  const hue2 = hueDegrees(adjustedA2, sample.b);
  const deltaLightness = sample.lightness - reference.lightness;
  const deltaChroma = adjustedChroma2 - adjustedChroma1;
  const hueDifference = hue2 - hue1;
  const absoluteHueDifference = Math.abs(hueDifference);
  const hueSum = hue1 + hue2;
  let deltaHue = 0;

  if (adjustedChroma1 * adjustedChroma2 !== 0) {
    if (absoluteHueDifference <= 180) deltaHue = hueDifference;
    else if (hueDifference > 180) deltaHue = hueDifference - 360;
    else deltaHue = hueDifference + 360;
  }

  const deltaAdjustedHue =
    2 * Math.sqrt(adjustedChroma1 * adjustedChroma2) * Math.sin((deltaHue * degreesToRadians) / 2);
  const meanLightness = (reference.lightness + sample.lightness) / 2;
  const meanAdjustedChroma = (adjustedChroma1 + adjustedChroma2) / 2;
  const meanAdjustedChroma7 = meanAdjustedChroma ** 7;
  let meanHue = hueSum;

  if (adjustedChroma1 * adjustedChroma2 !== 0) {
    if (absoluteHueDifference <= 180) meanHue = hueSum / 2;
    else if (hueSum < 360) meanHue = (hueSum + 360) / 2;
    else meanHue = (hueSum - 360) / 2;
  }

  const lightnessSquare = (meanLightness - 50) ** 2;
  const lightnessWeight = 1 + (0.015 * lightnessSquare) / Math.sqrt(20 + lightnessSquare);
  const chromaWeight = 1 + 0.045 * meanAdjustedChroma;
  const hueCrossTerm =
    1 -
    0.17 * Math.cos((meanHue - 30) * degreesToRadians) +
    0.24 * Math.cos(2 * meanHue * degreesToRadians) +
    0.32 * Math.cos((3 * meanHue + 6) * degreesToRadians) -
    0.2 * Math.cos((4 * meanHue - 63) * degreesToRadians);
  const hueWeight = 1 + 0.015 * meanAdjustedChroma * hueCrossTerm;
  const hueRotation = 30 * Math.exp(-(((meanHue - 275) / 25) ** 2));
  const chromaRotation = 2 * Math.sqrt(meanAdjustedChroma7 / (meanAdjustedChroma7 + twentyFive7));
  const rotationTerm = -Math.sin(2 * hueRotation * degreesToRadians) * chromaRotation;
  const lightnessTerm = deltaLightness / lightnessWeight;
  const chromaTerm = deltaChroma / chromaWeight;
  const hueTerm = deltaAdjustedHue / hueWeight;
  const differenceSquared = lightnessTerm ** 2 + chromaTerm ** 2 + hueTerm ** 2 + rotationTerm * chromaTerm * hueTerm;

  return Math.sqrt(Math.max(0, differenceSquared));
}

/** Return the perceptual CIEDE2000 difference between two #RRGGBB colors. */
export function paletteColorDistance(left: string, right: string) {
  const reference = hexToLab(left);
  const sample = hexToLab(right);
  if (!reference || !sample) return Number.POSITIVE_INFINITY;
  return deltaE2000(reference, sample);
}

export function arePaletteColorsNearDuplicates(left: string, right: string) {
  return paletteColorDistance(left, right) <= PALETTE_DELTA_E_THRESHOLDS.nearDuplicate;
}

/** Find the closest disjoint duplicate pairs, independent of palette order. */
export function findNearDuplicatePalettePairs(
  colors: readonly string[],
  options: { maxPairs?: number; threshold?: number } = {},
): PaletteSimilarityPair[] {
  const requestedMaxPairs = options.maxPairs ?? 2;
  const maxPairs = Number.isFinite(requestedMaxPairs) ? Math.max(0, Math.floor(requestedMaxPairs)) : 2;
  const requestedThreshold = options.threshold ?? PALETTE_DELTA_E_THRESHOLDS.nearDuplicate;
  const threshold = Number.isFinite(requestedThreshold)
    ? Math.max(0, requestedThreshold)
    : PALETTE_DELTA_E_THRESHOLDS.nearDuplicate;
  const candidates: Array<PaletteSimilarityPair & { leftIndex: number; rightIndex: number }> = [];

  for (let leftIndex = 0; leftIndex < colors.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < colors.length; rightIndex += 1) {
      const distance = paletteColorDistance(colors[leftIndex], colors[rightIndex]);
      if (!Number.isFinite(distance) || distance > threshold) continue;
      candidates.push({
        colors: [colors[leftIndex], colors[rightIndex]],
        distance,
        leftIndex,
        rightIndex,
      });
    }
  }

  candidates.sort(
    (left, right) =>
      left.distance - right.distance || left.leftIndex - right.leftIndex || left.rightIndex - right.rightIndex,
  );

  const usedIndexes = new Set<number>();
  const pairs: PaletteSimilarityPair[] = [];
  for (const candidate of candidates) {
    if (pairs.length >= maxPairs) break;
    if (usedIndexes.has(candidate.leftIndex) || usedIndexes.has(candidate.rightIndex)) continue;
    usedIndexes.add(candidate.leftIndex);
    usedIndexes.add(candidate.rightIndex);
    pairs.push({ colors: candidate.colors, distance: candidate.distance });
  }
  return pairs;
}

function bucketColor(bucket: PaletteBucket): Rgb {
  return {
    red: bucket.red / bucket.count,
    green: bucket.green / bucket.count,
    blue: bucket.blue / bucket.count,
  };
}

function clusterForBucket(bucket: PaletteBucket): PaletteCluster {
  return { bucket, lab: rgbToLab(bucketColor(bucket)) };
}

function nearestClusterIndex(color: LabColor, candidates: readonly PaletteCluster[]) {
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < candidates.length; index += 1) {
    const distance = deltaE2000(color, candidates[index].lab);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return { index: bestIndex, distance: bestDistance };
}

/**
 * Sample a deterministic 2D grid without the parity aliasing caused by a
 * fixed linear stride. Alternating quarter-cell offsets retain both sides of
 * common stripe and checkerboard patterns.
 */
function* sampledPixelIndexes(pixelCount: number, width: number, height: number, maxSamples: number) {
  if (pixelCount <= maxSamples) {
    for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) yield pixelIndex;
    return;
  }

  const imageWidth = Math.max(1, Math.floor(width));
  const imageHeight = Math.max(1, Math.min(Math.floor(height), Math.ceil(pixelCount / imageWidth)));
  const aspectRatio = imageWidth / imageHeight;
  const sampleColumns = Math.min(imageWidth, maxSamples, Math.max(1, Math.floor(Math.sqrt(maxSamples * aspectRatio))));
  const sampleRows = Math.min(imageHeight, Math.max(1, Math.floor(maxSamples / sampleColumns)));

  for (let row = 0; row < sampleRows; row += 1) {
    const horizontalOffset = row % 2 === 0 ? 0.25 : 0.75;
    for (let column = 0; column < sampleColumns; column += 1) {
      const verticalOffset = column % 2 === 0 ? 0.25 : 0.75;
      const sourceX = Math.min(imageWidth - 1, Math.floor(((column + horizontalOffset) * imageWidth) / sampleColumns));
      const sourceY = Math.min(imageHeight - 1, Math.floor(((row + verticalOffset) * imageHeight) / sampleRows));
      const pixelIndex = sourceY * imageWidth + sourceX;
      if (pixelIndex < pixelCount) yield pixelIndex;
    }
  }
}

export function extractPaletteFromRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  options: { maxColors?: number; maxSamples?: number } = {},
): ImagePaletteExtraction {
  const pixelCount = Math.min(Math.max(0, Math.floor(width) * Math.floor(height)), Math.floor(rgba.length / 4));
  const maxColors = Math.max(1, Math.min(16, Math.floor(options.maxColors ?? 8)));
  const maxSamples = Math.max(256, Math.floor(options.maxSamples ?? 16_384));
  const buckets = new Map<number, PaletteBucket>();
  let sampledPixels = 0;

  for (const pixelIndex of sampledPixelIndexes(pixelCount, width, height, maxSamples)) {
    const offset = pixelIndex * 4;
    if (rgba[offset + 3] < 128) continue;
    const red = rgba[offset];
    const green = rgba[offset + 1];
    const blue = rgba[offset + 2];
    const key = ((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4);
    const bucket = buckets.get(key) ?? { key, count: 0, red: 0, green: 0, blue: 0 };
    bucket.count += 1;
    bucket.red += red;
    bucket.green += green;
    bucket.blue += blue;
    buckets.set(key, bucket);
    sampledPixels += 1;
  }

  if (sampledPixels === 0) {
    return { swatches: [], sampledPixels: 0, representedPixels: 0, coverage: 0, confidence: 0 };
  }

  const ranked = [...buckets.values()]
    .sort((left, right) => right.count - left.count || left.key - right.key)
    .map(clusterForBucket);
  const selected: PaletteCluster[] = [];
  for (const cluster of ranked) {
    const nearest = nearestClusterIndex(cluster.lab, selected);
    if (nearest.distance <= PALETTE_DELTA_E_THRESHOLDS.extractionCluster) {
      const candidate = selected[nearest.index];
      candidate.bucket.count += cluster.bucket.count;
      candidate.bucket.red += cluster.bucket.red;
      candidate.bucket.green += cluster.bucket.green;
      candidate.bucket.blue += cluster.bucket.blue;
      candidate.lab = rgbToLab(bucketColor(candidate.bucket));
      continue;
    }
    if (selected.length < maxColors) selected.push(clusterForBucket({ ...cluster.bucket }));
  }

  const representedCounts = selected.map(() => 0);
  for (const cluster of ranked) {
    const nearest = nearestClusterIndex(cluster.lab, selected);
    if (nearest.distance <= PALETTE_DELTA_E_THRESHOLDS.representedCoverage) {
      representedCounts[nearest.index] += cluster.bucket.count;
    }
  }

  const analyzed = selected
    .map((cluster, index) => ({ bucket: cluster.bucket, representedCount: representedCounts[index] }))
    .sort(
      (left, right) =>
        right.representedCount - left.representedCount ||
        right.bucket.count - left.bucket.count ||
        left.bucket.key - right.bucket.key,
    );
  const representedPixels = analyzed.reduce((total, entry) => total + entry.representedCount, 0);
  const coverage = Math.min(100, (representedPixels / sampledPixels) * 100);
  const diversity = Math.min(1, analyzed.length / maxColors);
  const confidence = Math.min(0.99, 0.55 + (coverage / 100) * 0.35 + diversity * 0.04);

  return {
    swatches: analyzed.map(({ bucket, representedCount }) => ({
      color: rgbToHex(bucketColor(bucket)),
      sampleCount: representedCount,
      coverage: (representedCount / sampledPixels) * 100,
    })),
    sampledPixels,
    representedPixels,
    coverage,
    confidence,
  };
}

export function extractPaletteFromImage(image: HTMLImageElement, maxColors = 8) {
  const naturalWidth = image.naturalWidth || image.width;
  const naturalHeight = image.naturalHeight || image.height;
  if (naturalWidth < 1 || naturalHeight < 1) throw new Error('The selected image is not ready yet.');

  const maxEdge = 160;
  const scale = Math.min(1, maxEdge / Math.max(naturalWidth, naturalHeight));
  const width = Math.max(1, Math.round(naturalWidth * scale));
  const height = Math.max(1, Math.round(naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Color extraction is not supported in this browser.');
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height);
  return extractPaletteFromRgba(pixels.data, width, height, { maxColors });
}
