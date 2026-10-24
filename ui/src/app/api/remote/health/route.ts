import { getCloudflaredStatus } from '@/server/cloudflared';
import { getOllamaStatus } from '@/server/ollama';
import { getExecutionHealth } from '@/server/executionHealth';


export async function GET() {
  return Response.json({
    ok: true,
    app: 'ai-toolkit',
    timestamp: new Date().toISOString(),
    ...await getExecutionHealth(),
    cloudflared: await getCloudflaredStatus(),
    ollama: await getOllamaStatus(),
  });
}
