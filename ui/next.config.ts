import type { NextConfig } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';

function getAppVersion() {
  try {
    const versionFile = readFileSync(join(process.cwd(), '..', 'version.py'), 'utf8');
    const versionMatch = versionFile.match(/VERSION\s*=\s*["']([^"']+)["']/);
    return versionMatch ? versionMatch[1] : 'unknown';
  } catch {
    return 'unknown';
  }
}

const nextConfig: NextConfig = {
  distDir: process.env.AITK_NEXT_DIST_DIR || '.next',
  env: {
    NEXT_PUBLIC_APP_VERSION: getAppVersion(),
  },
  serverExternalPackages: [
    'archiver',
    'macstats',
    'node-cache',
    'osx-temperature-sensor',
    'sharp',
    'systeminformation',
    'yauzl',
  ],
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('osx-temperature-sensor', 'macstats');
    }
    return config;
  },
  devIndicators: false,
  experimental: {
    webpackBuildWorker: true,
    serverActions: {
      bodySizeLimit: '2mb',
    },
    middlewareClientMaxBodySize: '5gb',
  },
};

export default nextConfig;
