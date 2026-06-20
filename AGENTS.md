# Agent Instructions

## Project Shape

- AITK Studio is a Python diffusion-model training toolkit with a Next.js/React control surface in `ui/`.
- Root-level Python code (`run.py`, `toolkit/`, `extensions/`, `extensions_built_in/`) owns training, model integration, and CLI workflows.
- `ui/` is a Next 15 / React 19 / TypeScript app with Tailwind, Prisma-backed UI state, API routes under `ui/src/app/api/**/route.ts`, server helpers under `ui/src/server/`, shared helpers under `ui/src/utils/`, and a cron worker under `ui/cron/`.
- When changing code that touches datasets, configs, runs, outputs, models, project files, or training/generation state, preserve project-space isolation as well as global-path behavior. Use existing project helpers such as `resolveDatasetScope`, `getProjectRoots`, and `ensureProjectFolders` instead of hard-coded global folders so project-scoped data stays under `PROJECTS_FOLDER`.
- Treat `datasets/`, `models/`, `output/`, `projects/`, `aitk_db.db*`, `.tmp/`, and generated runtime artifacts as user/runtime data. Do not clean, rename, or rewrite them unless the task explicitly requires it.
- Do not hand-edit generated Prisma client files under `ui/src/generated/prisma/`; update the Prisma/schema path and regenerate instead.

## Local Commands

- UI commands run from `ui/`: `npm run dev`, `npm run build`, `npm run start`, and the targeted `npm run test:*` scripts in `ui/package.json`.
- `npm run dev` starts the managed app stack on port `3000`: Next UI, cron worker, and updater.
- `npm run start` starts the managed app stack on port `8675` after DB prep; TensorBoard may use port `6006` when enabled.
- `npm run build` compiles the worker with `tsconfig.worker.json` and then runs `next build`. Because `next.config.ts` currently has `typescript.ignoreBuildErrors: true`, do not treat a Next build alone as proof that all UI route/page types are clean.
- Prefer the narrowest relevant verification: run the matching `npm run test:<area>` script for touched UI/server utilities, and use `python -m py_compile` or focused Python tests/checks for Python-only changes.
- After completing any local server-based testing, stop all servers and background helper processes started for the test before handing work back to the user. Verify the relevant localhost ports or processes are closed when practical.

## TypeScript Guidance

- Keep `strict` TypeScript expectations intact. Do not weaken compiler settings or add broad casts to get past type errors.
- Prefer `unknown` at runtime boundaries: `request.json()`, `response.json()`, filesystem YAML/JSON, DB JSON strings, environment variables, remote worker responses, Ollama/OpenRouter/Hugging Face/Comfy responses, and browser extension APIs. Narrow with local type guards, validators, or existing `clean*` / `normalize*` helpers before use.
- Avoid introducing new `any`. If legacy code already uses `any`, contain it near the boundary and convert to named types, `Record<string, unknown>`, or a narrowed shape as soon as practical.
- Let inference handle obvious local variables. Add explicit types for exported functions, reusable helpers, React props, API response shapes, and cross-module contracts.
- Prefer `satisfies` for option/config maps when the object should be checked against a shape without losing literal inference.
- Derive unions from constants with `as const` and `(typeof values)[number]` for statuses, modes, providers, and UI option values. Avoid duplicating runtime arrays and TypeScript unions.
- Prefer literal unions and `as const` objects over `enum` for serializable values that cross API, JSON, DB, or config boundaries.
- Model async UI and worker states as discriminated unions when a loose bag of optional properties would allow impossible states.
- Use exhaustive `never` checks for switches over discriminated unions or fixed status/mode/provider sets.
- Build new types from existing domain types with `Pick`, `Omit`, `Partial`, indexed access types, and utility types instead of duplicating object shapes.
- Remember that type-safe is not runtime-safe: casts like `as JobConfig` or generic calls such as `remoteJson<Job>()` still need validation when data comes from outside the current process.

## UI and API Patterns

- Keep route handlers thin: parse and validate inputs in `ui/src/app/api/**/route.ts`, then delegate durable logic to `ui/src/server/` or pure utilities in `ui/src/utils/`.
- Keep shared business logic outside React components when it can be tested with the existing Node test scripts.
- Follow the existing UI style: Tailwind utilities, `classNames`, existing components/modals/hooks, and established dark theme tokens.
- For icons, prefer the icon libraries already in use (`lucide-react` or `react-icons`) instead of custom inline SVG unless the existing component requires it.
- Next route params use the project's current Next 15 style, often typed as `Promise<{ ... }>` in route handlers.

## Security and Data Handling

- Remote workers, encrypted datasets, bearer auth, OpenRouter/Ollama calls, Hugging Face downloads, Cloudflared, and update flows are security-sensitive. Do not log or persist dataset secrets, API tokens, bearer tokens, unwrapped encryption keys, or plaintext encrypted-dataset contents unless existing code already deliberately does so.
- Preserve path-scope checks for datasets, projects, files, and remote imports. Prefer existing helpers such as dataset/project scope resolvers instead of ad hoc path concatenation.
- Treat network, filesystem, subprocess, and database inputs as untrusted even when TypeScript types say otherwise.
