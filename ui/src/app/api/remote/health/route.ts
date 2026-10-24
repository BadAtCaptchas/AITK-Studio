import { getCloudflaredStatus } from '@/server/cloudflared';
import { getOllamaStatus } from '@/server/ollama';
import { getExecutionHealth } from '@/server/executionHealth';
import { continuationCapabilities } from '@/server/loraContinuation';


export async function GET() {
  return Response.json({
    ok: true,
    app: 'ai-toolkit',
    loraContinuation: continuationCapabilities,
    timestamp: new Date().toISOString(),
    ...await getExecutionHealth(),
    cloudflared: await getCloudflaredStatus(),
    ollama: await getOllamaStatus(),
  });
}
