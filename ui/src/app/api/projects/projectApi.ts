import { NextResponse } from 'next/server';
import { ProjectError } from '@/server/projects';
import { isRequestAuthenticated } from '@/utils/authSession';

export async function ensureProjectApiAccess(request: Request): Promise<NextResponse | null> {
  const expectedToken = process.env.AI_TOOLKIT_AUTH;
  if (!(await isRequestAuthenticated(request, expectedToken))) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

export function projectApiError(error: unknown, fallback: string) {
  if (error instanceof ProjectError) {
    return NextResponse.json(
      {
        error: error.message,
        code: error.code,
        ...(error.details === undefined ? {} : { details: error.details }),
      },
      { status: error.status },
    );
  }
  console.error(fallback, error);
  return NextResponse.json({ error: fallback }, { status: 500 });
}

export async function readJsonObject(request: Request): Promise<Record<string, unknown>> {
  const value: unknown = await request.json();
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ProjectError('Request body must be a JSON object', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  return value as Record<string, unknown>;
}
