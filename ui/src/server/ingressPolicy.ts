import net from 'net';
import os from 'os';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
let tunnelCache: { until: number; url: string } | null = null;
function quickTunnelURL(): string {
  if (tunnelCache && tunnelCache.until > Date.now()) return tunnelCache.url;
  let url = '';
  try {
    const filename = path.join(TOOLKIT_ROOT, '.cloudflared.url');
    if (fs.statSync(filename).size < 2048) {
      const value = fs.readFileSync(filename, 'utf8').trim();
      if (/^https:\/\/[a-z0-9-]+\.trycloudflare\.com\/?$/.test(value)) url = value;
    }
  } catch {
    /* Tunnel is not running yet. */
  }
  tunnelCache = { until: Date.now() + 1000, url };
  return url;
}
import type { IncomingMessage, OutgoingHttpHeaders } from 'http';

function address(value: string): string {
  return value
    .replace(/^::ffff:/, '')
    .replace(/^\[|\]$/g, '')
    .toLowerCase();
}
function loopback(value: string): boolean {
  return value === 'localhost' || value === '::1' || /^127\./.test(value);
}
export function ingressConfig(env: NodeJS.ProcessEnv = process.env) {
  const host = address(env.AITK_BIND_HOST?.trim() || '127.0.0.1');
  if (host !== 'localhost' && net.isIP(host) === 0)
    throw new Error('AITK_BIND_HOST must be an IP address or localhost');
  const network = !loopback(host);
  if (network && (env.AITK_NETWORK_MODE !== '1' || !env.AI_TOOLKIT_AUTH?.trim())) {
    throw new Error('Network binding requires AITK_NETWORK_MODE=1 and AI_TOOLKIT_AUTH. Default startup uses loopback.');
  }
  let publicUrl: URL | undefined;
  const tunnelEnabled = /^(1|true)$/i.test(env.AITK_CLOUDFLARED_ENABLED || '');
  if (tunnelEnabled && !env.AI_TOOLKIT_AUTH?.trim()) throw new Error('Cloudflared requires AI_TOOLKIT_AUTH');
  const configuredURL =
    env.AITK_PUBLIC_URL || (tunnelEnabled ? env.AITK_CLOUDFLARED_PUBLIC_URL || quickTunnelURL() : '');
  if (configuredURL) {
    publicUrl = new URL(configuredURL);
    if (!['http:', 'https:'].includes(publicUrl.protocol) || publicUrl.username || publicUrl.password)
      throw new Error('Invalid AITK_PUBLIC_URL');
  }
  const trusted = new Set(
    (env.AITK_TRUSTED_PROXY_IPS || '')
      .split(',')
      .map(value => address(value.trim()))
      .filter(Boolean),
  );
  if ([...trusted].some(value => net.isIP(value) === 0))
    throw new Error('AITK_TRUSTED_PROXY_IPS must contain exact IP addresses');
  const allowed = new Set(['localhost', '127.0.0.1', '::1', host, os.hostname().toLowerCase()]);
  if (network)
    for (const entries of Object.values(os.networkInterfaces()))
      for (const entry of entries || []) allowed.add(address(entry.address));
  if (publicUrl) allowed.add(address(publicUrl.hostname));
  for (const value of (env.AITK_ALLOWED_HOSTS || '').split(',')) if (value.trim()) allowed.add(address(value.trim()));
  return { host, network, publicUrl, trusted, allowed, authenticated: Boolean(env.AI_TOOLKIT_AUTH?.trim()) };
}

export function browserRequestError(headers: Headers, url: string, method: string): string | null {
  const policy = ingressConfig();
  let target: URL;
  try {
    target = new URL(`http://${headers.get('host') || new URL(url).host}`);
  } catch {
    return 'Invalid Host';
  }
  if (!policy.allowed.has(address(target.hostname))) return 'Host is not allowed';
  const origin = headers.get('origin');
  if (origin) {
    try {
      const expected =
        (policy.publicUrl?.host === target.host ? policy.publicUrl.origin : null) ||
        `${headers.get('x-aitk-forwarded-proto') === 'https' ? 'https:' : new URL(url).protocol}//${target.host}`;
      if (new URL(origin).origin !== expected) return 'Origin is not allowed';
    } catch {
      return 'Invalid Origin';
    }
  }
  if (headers.get('sec-fetch-site') === 'cross-site') return 'Cross-site requests are not allowed';
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method) && headers.has('cookie') && !origin && !headers.has('authorization'))
    return 'Origin is required for browser commands';
  return null;
}

export function trustedForwardHeaders(request: IncomingMessage): OutgoingHttpHeaders {
  const policy = ingressConfig();
  const peer = address(request.socket.remoteAddress || '');
  const headers = { ...request.headers };
  for (const name of Object.keys(headers)) {
    if (name.startsWith('x-aitk-') && ['x-aitk-client-ip', 'x-aitk-forwarded-proto'].includes(name))
      delete headers[name];
    if (name === 'forwarded' || name.startsWith('x-forwarded-')) delete headers[name];
  }
  let client = peer;
  let protocol = 'http';
  if (policy.trusted.has(peer)) {
    const chain = String(request.headers['x-forwarded-for'] || '')
      .split(',')
      .map(value => address(value.trim()))
      .filter(value => net.isIP(value));
    while (chain.length && policy.trusted.has(client)) client = chain.pop()!;
    if (String(request.headers['x-forwarded-proto']).split(',')[0].trim() === 'https') protocol = 'https';
  }
  headers['x-aitk-client-ip'] = client;
  headers['x-aitk-forwarded-proto'] =
    policy.publicUrl && policy.publicUrl.host === request.headers.host && policy.publicUrl.protocol === 'https:'
      ? 'https'
      : protocol;
  return headers;
}
