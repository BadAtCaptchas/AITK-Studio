# AITK Studio UI

Train models, manage datasets, and generate images in AITK Studio.

## Getting Started

From the `ui` directory, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

Production builds use:

```bash
npm run build
npm run start
```

`npm run dev` and `npm run start` also run `npm run update_db` so the local database schema is prepared before the UI starts.

For everyday use, build once after installing or updating, then use `npm run start` to reopen the app at [http://localhost:8675](http://localhost:8675). Starting an existing build does not compile the interface again.

To open Studio in its own window, use your browser's **Install app** or **Open as app** command. The training service must remain running while you use Studio.

## Bug Reports

Report reproducible UI bugs in the main repository: [github.com/BadAtCaptchas/AITK-Studio/issues/new?template=bug_report.md](https://github.com/BadAtCaptchas/AITK-Studio/issues/new?template=bug_report.md).

## Useful Scripts

- `npm run update_db` prepares the database and Prisma client.
- `npm run build` builds the worker, API server, and interface.
- `npm run start` starts the built worker and UI on port `8675`.
- `npm run test:metrics`, `npm run test:advisor`, `npm run test:tensorboard`, `npm run test:telemetry`, `npm run test:remote`, and `npm run test:scripts` run focused Node tests.

## Framework Reference

See [Vite](https://vite.dev/guide/), [React Router](https://reactrouter.com/start/declarative/routing), and [Hono](https://hono.dev/docs/).
