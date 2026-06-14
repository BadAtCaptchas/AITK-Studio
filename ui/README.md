# AITK Studio UI

This is the Next.js UI for `BadAtCaptchas/AITK-Studio`. It provides the web interface, local job worker, database preparation scripts, and supporting UI tooling for this fork.

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

## Bug Reports

Report reproducible UI bugs in the main repository: [github.com/BadAtCaptchas/AITK-Studio/issues/new?template=bug_report.md](https://github.com/BadAtCaptchas/AITK-Studio/issues/new?template=bug_report.md).

## Useful Scripts

- `npm run update_db` prepares the database and Prisma client.
- `npm run build` builds the worker and Next.js app.
- `npm run start` starts the built worker and UI on port `8675`.
- `npm run test:metrics`, `npm run test:advisor`, `npm run test:tensorboard`, `npm run test:remote`, and `npm run test:scripts` run focused Node tests.

## Framework Reference

For framework-level documentation, see the [Next.js documentation](https://nextjs.org/docs).
