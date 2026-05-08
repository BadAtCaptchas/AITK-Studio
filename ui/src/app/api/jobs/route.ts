import { NextResponse } from 'next/server';
import { isMac } from '@/helpers/basic';
import { db } from '@/server/db';

function isValidJobName(name: unknown) {
  if (typeof name !== 'string' || name.trim().length === 0) {
    return false;
  }

  if (name === '.' || name.includes('..')) {
    return false;
  }

  return name === name.split('/').pop() && name === name.split('\\').pop();
}


export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');

  try {
    if (id) {
      const job = await db.jobs.findById(id);
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await db.jobs.findLatestByRef(job_ref);
      return NextResponse.json(job);
    }

    const jobs = await db.jobs.list({ job_type });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config } = body;

    if (!isValidJobName(name)) {
      return NextResponse.json({ error: 'Invalid job name' }, { status: 400 });
    }
    let gpu_ids: string = body.gpu_ids;

    if (isMac()) {
      gpu_ids = "mps";
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }

    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id && typeof id !== 'string') {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    if (id) {
      // Update existing training
      const training = await db.jobs.update(id, {
        name,
        gpu_ids,
        job_config: JSON.stringify(job_config),
        ...extra,
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;

      // Create new training
      const training = await db.jobs.create({
        name,
        gpu_ids,
        job_config: JSON.stringify(job_config),
        queue_position: newQueuePosition,
        ...extra,
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
