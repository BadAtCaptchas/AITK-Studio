import fs from 'node:fs';
import path from 'node:path';
import { parse } from 'dotenv';
import { expand } from 'dotenv-expand';

/** Keep shell values authoritative; preserve the application's env-file precedence. */
export function loadEnvironment(directory, mode, environment = process.env) {
  const names = [`.env.${mode}.local`, ...(mode === 'test' ? [] : ['.env.local']), `.env.${mode}`, '.env'];
  const parsed = {};
  for (const name of names) {
    let values;
    try { values = parse(fs.readFileSync(path.join(directory, name))); }
    catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    for (const [key, value] of Object.entries(values)) if (!(key in parsed)) parsed[key] = value;
  }
  expand({ parsed, processEnv: environment });
}
