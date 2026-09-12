# Execution and recovery contract

The control plane retains SQLite/MongoDB and the single global workspace. Schema changes are additive; existing output directories are never moved by startup.

## Ownership

- A job owns at most one current attempt. A conditional status/attempt update wins before preparation or spawning. Duplicate commands conflict or attach to the accepted operation.
- Attempts carry immutable configuration and an opaque generation ID into Python. Both Python UIJobStore and Node callbacks must include that generation when updating status or consuming controls.
- A worker's device reservation is one compare-and-swap record containing all reserved physical devices. Device overlap uses a sorted set; CUDA launch order remains the user's order. No partial multi-device claims.
- Resource ownership does not expire merely because Next or cron restarts. Recovery checks the owning attempt/process, and stale writers cannot publish into a newer generation.
- Existing jobs keep their original storage key. Display names can change without relocating outputs. Configuration snapshots contain no supplied dataset keys.

## Durable state

A versioned RuntimeRecord provides compare-and-swap storage for attempts, operations, reservations, sessions and synchronization cursors. Values are bounded JSON and validated by each domain. SQLite uses conditional updates and MongoDB uses atomic single-document operations. Separate stores use explicit key namespaces and never expose their contents through settings APIs.

Remote operations persist safe phase/identity information before acknowledging a command. Supplied ephemeral dataset keys remain in memory: after restart these operations wait for keys again. Durable encrypted resume continues to require the existing explicit opt-in and wrapping secret. Remote mutations use stable idempotency identities and reconciliation before retrying an uncertain result.

## Upgrade and rollback

Preparation is exclusive and version/checksum guarded. No-op starts do not rotate backups. A named pre-upgrade SQLite online snapshot is retained outside routine retention. MongoDB migration ownership is shared through the database. A failed migration remains incomplete and startup fails closed. No force-reset, accept-data-loss, or runtime-folder cleanup is used.

Rollback requires stopping the app and all affected attempts before restoring a database snapshot and its matching application revision. Do not run older control-plane code against new active attempts: older writers do not implement fencing.

## Compatibility

Mutating GET commands return 405; update controllers and workers together. Legacy browser cookies and old broadly scoped asset signatures expire at the upgrade. Machine bearer credentials remain independent. Direct installations bind loopback by default; network/container operation requires explicit authenticated network mode. Readiness reports the selected runtime and deployment policy.

## Operator setup

Use Node 22.12+, 24 or 26; CI covers Node 22/24/26 across Linux and Windows. Use the reference Python 3.12 environment (3.11 for DGX). Run `python scripts/install_runtime.py` inside a virtual environment to resolve Torch, TorchCodec and application dependencies together using the default blackwell-cu130 profile (CUDA 13.0, requiring a compatible NVIDIA driver). `--dry-run --report resolution.json` performs fresh resolution without changing the environment. Use `--profile legacy-cu128` or `--profile blackwell-cu128` for explicit CUDA 12 compatibility; DGX and macOS require `--profile dgx-cu130` and `--profile macos`, respectively. Do not follow installation with unpinned Torch/Transformers upgrades.

`AITK_PYTHON_PATH` explicitly selects an interpreter. Otherwise `.venv` takes precedence over `venv`, then the active environment and system command. A broken preferred environment is an error. Use `python scripts/environment_doctor.py --arch flux` or Settings / Runtime diagnostics to check real imports, versions, CUDA/MPS, binary compatibility, package dependencies, FFmpeg, storage and worker freshness. Diagnostics do not download weights or launch training. Model discovery is lazy; `python scripts/check_model_imports.py` deliberately imports every declared public class and fails on broken integrations.

For LAN/container access, set `AITK_BIND_HOST=0.0.0.0`, `AITK_NETWORK_MODE=1` and a nonempty `AI_TOOLKIT_AUTH`. For a reverse proxy set the exact external `AITK_PUBLIC_URL`, and list only the proxy's exact IPs in `AITK_TRUSTED_PROXY_IPS` when forwarding client identity. `AITK_ALLOWED_HOSTS` adds explicit hostnames. Untrusted forwarding headers are stripped. Named Cloudflared tunnels use `AITK_CLOUDFLARED_PUBLIC_URL`; quick tunnels accept only the exact generated hostname from `.cloudflared.url`. Cloudflared requires authentication. Local and configured public origins can each be used without weakening origin checks.

Browser sessions are random opaque handles, stored hashed, with a 24-hour absolute and 2-hour idle limit. Logout revokes the current handle; authenticated DELETE `/api/auth?all=1` revokes all browser sessions. Existing deterministic cookies require a new login. Bearer tokens are machine credentials and do not become browser cookies.

JSON commands are limited to 2 MiB (login: 4 KiB), 15 seconds, depth 32 and bounded collections. The front server enforces byte limits before Next middleware clones a body. Mutations use POST/DELETE; old GET command URLs return 405. Controller and workers must be upgraded together. The installed Next 15 middleware compatibility patch backports [the upstream request-body finalization fix](https://github.com/vercel/next.js/pull/85418); the patch checks the expected implementation and fails closed if it changes.

## Upload and operation recovery

Archives use chunks of at most 64 MiB, with a stable upload ID and identical retry bytes. A direct binary archive request has the same 64 MiB limit; larger archives must use the chunk protocol. There are at most 8 active uploads, 128 GiB of combined staged/reserved bytes, and a 2 GiB free-space reserve. Manifests freeze at finalization, verify every chunk hash and reject conflicting parameters. Extraction is private, bounded and rejects links, traversal, duplicate files and Windows-special paths.

An accepted operation returns HTTP 202 with an operation ID. GET `/api/operations/<id>` reports durable state and whether cancellation is still available. DELETE requests cancellation before publication or remote launch begins. After that atomic boundary it returns 409; wait for confirmation and use the normal job stop/delete flow. `needs-keys` means the ephemeral unlock handoff expired or the stack restarted; unlock again to resume. No plaintext dataset keys are persisted in operation records.

The worker resumes queued or interrupted operations and deduplicates remote imports/starts by stable IDs. It never steals a live owner's lease based on heartbeat age. If host identity or process birth cannot be verified, ownership remains blocked for inspection. Caption merging uses the same process/attempt ownership rules and can recover an interrupted merge. Do not remove ownership records while their recorded process is alive.

## Verification and limits

CI includes strict app/worker types, lint, serial Node behavior tests, fresh production browser journeys, an installed production-only stack, SQLite/Mongo adapter races, shared Python/TypeScript configuration fixtures, Python behavior tests and real model-class imports. Hardware-profile jobs retain pip resolver reports. The browser fixture owns an isolated temporary database/storage root and stops its services after testing.

These checks do not measure training throughput, memory usage, optimizer correctness, or generated quality. CUDA/Blackwell/DGX/macOS execution still needs the matching physical hardware. The configuration review identifies worker/device, model artifact, and image/video/audio output; available memory is not a measured training-memory guarantee.
