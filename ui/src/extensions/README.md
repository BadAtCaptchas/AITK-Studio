# Model UI extensions

Installed Python extension packages in `extensions/` or `extensions_built_in/`
can include `ui.tsx`, `ui.ts`, `ui.jsx`, or `ui.js`. Export an
`AI_TOOLKIT_UI_MODELS` array of `ModelArch` entries (see `domain/modelOptions.ts`).
Studio reloads these on navigation without rebuilding the UI. Custom packages
override built-in entries by model name; built-in Studio defaults remain available
if a module fails. Only install trusted packages: their UI code runs in the browser,
just as their Python code runs in the trainer.

Supported imports: React, React JSX runtimes, `next/link`,
`@/helpers/defaultSamples`, `@/components/formInputs`, `@/types`, and
`@/app/jobs/new/options`. `customSections(config, setJobConfig)` can return React
controls; `modelNotes` may be JSX. No arbitrary imports are resolved.

For custom Python architectures, the job config must declare them under a namespaced
`extensions` entry, for example
`extensions: { "mycompany.models": { modelArches: ["custom_arch"] } }`.
The training form adds a declaration when selecting an installed custom model.
