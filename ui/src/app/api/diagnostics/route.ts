import { getReadiness } from '@/server/readiness';
export async function GET() {
  try {
    return Response.json(await getReadiness(), { headers: { 'Cache-Control': 'no-store' } });
  } catch {
    return Response.json(
      { error: 'Diagnostics unavailable. Check database and worker startup logs.' },
      { status: 503 },
    );
  }
}
