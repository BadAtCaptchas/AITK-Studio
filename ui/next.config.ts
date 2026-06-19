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
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    webpackBuildWorker: true,
    serverActions: {
      bodySizeLimit: '5gb',
    },
    middlewareClientMaxBodySize: '5gb',
  },
};

export default nextConfig;
