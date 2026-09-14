# Deployment and operations

[← Back to AITK Studio](../README.md#documentation)

Run AITK Studio on a server, connect workers, and manage storage and monitoring. Shell examples assume Linux unless labeled otherwise; npm commands run from `ui`, while Docker Compose commands run from the repository root. Read [execution and recovery](execution-safety.md) before upgrading an active installation.

- [Security](#security)
- [Docker Compose](#docker-compose)
- [TensorBoard](#tensorboard)
- [UI database](#ui-database)
- [Remote workers and Cloudflare Tunnel](#remote-workers-and-cloudflare-tunnel)
- [Secure remote Ollama captioning](#secure-remote-ollama-captioning)
- [RunPod](#runpod)
- [Modal](#modal)
- [Upgrading from workspace releases](#upgrading-from-workspace-releases)

## Security

Direct installations bind to localhost by default. To allow LAN or server access, set an explicit network mode and a strong bearer token before launching the UI from the `ui` directory.

Linux:

```bash
AITK_BIND_HOST=0.0.0.0 AITK_NETWORK_MODE=1 AI_TOOLKIT_AUTH="replace-with-a-strong-token" npm run build_and_start
```

Windows PowerShell:

```powershell
$env:AITK_BIND_HOST="0.0.0.0"
$env:AITK_NETWORK_MODE="1"
$env:AI_TOOLKIT_AUTH="replace-with-a-strong-token"
npm run build_and_start
```

Windows Command Prompt:

```bat
set AITK_BIND_HOST=0.0.0.0
set AITK_NETWORK_MODE=1
set AI_TOOLKIT_AUTH=replace-with-a-strong-token
npm run build_and_start
```

With `AI_TOOLKIT_AUTH` set, API routes require a matching bearer token or authenticated browser session; unauthenticated calls are rejected with `401 Unauthorized`. This includes job creation and queue/start endpoints.

For a reverse proxy, set the exact external `AITK_PUBLIC_URL` and only the proxy's exact IPs in `AITK_TRUSTED_PROXY_IPS`. See the [operator setup contract](execution-safety.md#operator-setup) for allowed hosts, tunnel origins, browser sessions, and recovery.

## Docker Compose

The Compose deployment builds this checkout by default, so the running image always matches the fork you are working from:

```bash
AI_TOOLKIT_AUTH="replace-with-a-strong-token" \
GIT_COMMIT="$(git rev-parse HEAD)" \
docker compose up --build
```

Set `AITK_IMAGE` to use a prebuilt image instead, then run `docker compose pull` and `docker compose up --no-build`. Avoid floating tags for production deployments.

Compose persists `data/`, `datasets/`, `output/`, `config/`, `models/` on the host. SQLite is stored at `data/aitk_db.db`; using a directory mount avoids Docker creating a directory when a database file does not exist yet. If you used an older version of this Compose file, move the existing database once before starting the new deployment:

```bash
mkdir -p data
cp aitk_db.db data/aitk_db.db
```

## TensorBoard

TensorBoard is installed with the Python requirements. If `AITK_ENABLE_TENSORBOARD` is not set, the UI tries to auto-enable TensorBoard when the package is available in the active Python environment and silently skips it if the probe or startup fails.

You can force it on or off when starting the UI:

```bash
# Linux/macOS
AITK_ENABLE_TENSORBOARD=1 npm run build_and_start
AITK_ENABLE_TENSORBOARD=0 npm run build_and_start

# Windows PowerShell
$env:AITK_ENABLE_TENSORBOARD="1"; npm run build_and_start
$env:AITK_ENABLE_TENSORBOARD="0"; npm run build_and_start
```

When TensorBoard is enabled, the UI starts it on port `6006`, writes a small `aitk_status` run so TensorBoard has data before the first training job, writes UI-launched training events to `<training folder>/.tensorboard`, and shows a TensorBoard link on the dashboard and train job overview.

TensorBoard is a separate service and is not protected by `AI_TOOLKIT_AUTH`; use localhost binding, a firewall, or proxy auth when exposing it outside a trusted network.

Optional environment variables:

- `AITK_TENSORBOARD_PORT=6006` changes the TensorBoard port.
- `AITK_TENSORBOARD_HOST=0.0.0.0` changes the bind host.
- `AITK_TENSORBOARD_LOG_DIR=/path/to/logs` changes the event log directory.
- `AITK_TENSORBOARD_PUBLIC_URL=http://host:6006` changes the link shown in the UI, useful behind proxies or custom Docker port mappings.
- `AITK_TENSORBOARD_STATUS_RUN=0` removes and stops writing the synthetic `aitk_status` run. Without another run, TensorBoard may show an empty dashboard until training writes events.

For Docker Compose, leave `AITK_ENABLE_TENSORBOARD` unset for auto-detection, or set it explicitly. The Compose file binds the published TensorBoard port to host `127.0.0.1` by default, even though TensorBoard listens on `0.0.0.0` inside the container so Docker port forwarding can reach it:

```bash
AITK_ENABLE_TENSORBOARD=1 docker compose up
AITK_ENABLE_TENSORBOARD=0 docker compose up
```

## UI database

The UI uses SQLite by default and stores its state in `aitk_db.db`. You can switch all UI-backed state to MongoDB at startup:

```bash
AITK_DB_PROVIDER=mongodb \
AITK_MONGODB_URI="mongodb://localhost:27017" \
AITK_MONGODB_DB=ai_toolkit \
npm run build_and_start
```

Supported database environment variables:

- `AITK_DB_PROVIDER=sqlite|mongodb` defaults to `sqlite`.
- `AITK_SQLITE_PATH` defaults to `../aitk_db.db` from the `ui` folder.
- `AITK_SQLITE_BACKUP_RETENTION` controls how many consistent pre-migration SQLite backups are retained (default `3`; set `0` to disable).
- `AITK_MONGODB_URI` is required when `AITK_DB_PROVIDER=mongodb`.
- `AITK_MONGODB_DB` defaults to `ai_toolkit`.

Run `npm run update_db` after changing database settings. SQLite mode prepares Prisma and the SQLite schema. MongoDB mode prepares the MongoDB indexes while still generating the Prisma client for SQLite fallback support.

To migrate existing SQLite UI data into MongoDB, leave `aitk_db.db` and the training output folders in place, set the MongoDB variables, then run:

```bash
cd ui
AITK_MONGODB_URI="mongodb://localhost:27017" npm run migrate_sqlite_to_mongo
```

The migration imports jobs, queues, settings, and existing per-job `loss_log.db` metrics. SQLite files are left untouched so you can switch back to SQLite.

## Remote workers and Cloudflare Tunnel

The UI can control remote AITK Studio worker instances. Each worker runs the same UI/cron app with `AI_TOOLKIT_AUTH` set. Add the worker from **Settings → Compute** using its public URL and bearer token. When you start a job assigned to a remote worker, the central UI creates a `.aitk.zip` job bundle with datasets, uploads it to the worker, starts the worker queue, and then proxies logs, metrics, samples, checkpoints, and exports back through the central UI.

Remote workers are authoritative after upload. The central UI mirrors status, step, speed, config, and error text from the worker. Base model files are not bundled; they must exist on the worker or the import will report warnings.

Encrypted dataset bundles include only ciphertext dataset folders. Starting an encrypted job on a remote worker requires supplying the dataset secret at start time unless durable encrypted resume was enabled for that job. YubiKey-protected encrypted datasets use the same central unlock path: the browser connected to the central UI prompts for the USB security key, unwraps the dataset key, and the central server forwards only the ephemeral dataset key to the HTTPS worker start request. Remote workers do not need a YubiKey, USB access, or a native FIDO helper. Remote encrypted starts require an `https://` worker URL unless `AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS=1` is set explicitly.

Optional managed `cloudflared` support is configured with environment variables on any instance you want to expose through Cloudflare Tunnel:

```bash
AITK_CLOUDFLARED_ENABLED=1
AITK_CLOUDFLARED_PUBLIC_URL=https://your-worker.example.com
AITK_CLOUDFLARED_TOKEN_FILE=/path/to/cloudflared-token
AITK_CLOUDFLARED_TARGET_URL=http://127.0.0.1:8675
AITK_CLOUDFLARED_METRICS_ADDR=127.0.0.1:60123
AITK_CLOUDFLARED_LOG_LEVEL=info
AITK_CLOUDFLARED_AUTO_DOWNLOAD=0
```

`AI_TOOLKIT_AUTH` is required when `AITK_CLOUDFLARED_ENABLED=1`. `AITK_CLOUDFLARED_TOKEN_FILE` is optional: when it is set, the app starts a named tunnel with that token; when it is not set, the app starts a Cloudflare quick tunnel with a random `trycloudflare.com` URL and shows the generated URL in Settings after Cloudflared reports it. `AITK_CLOUDFLARED_PUBLIC_URL` is optional metadata for named tunnels. The app can start, stop, download, and show tunnel status from the Settings page; Docker images include `cloudflared` for this workflow. If the binary is missing and no custom `AITK_CLOUDFLARED_BIN` is set, the Settings page can download the official Cloudflare GitHub release into `bin/cloudflared` (`bin/cloudflared.exe` on Windows). Set `AITK_CLOUDFLARED_AUTO_DOWNLOAD=1` or enable the Settings checkbox to download automatically before starting.

## Secure remote Ollama captioning

The Queue page includes a **Secure Remote Captioning** job view for image datasets. It starts a local UI caption job that streams one image at a time to a selected **Remote Ollama** endpoint. The remote host can be a standalone Ollama server; it does not need to run the AITK Studio UI. Add direct endpoints from **Settings > Remote Ollama** with the Ollama base URL, such as `http://ollama-host:11434`, and an optional bearer token for protected reverse proxies or tunnels.

Direct Remote Ollama sends prompt, optional system prompt, and image bytes to the configured Ollama HTTP API. Use HTTPS or a protected reverse proxy outside a trusted LAN. The dataset is not bundled or stored on the remote host. The optional system prompt is saved per dataset in the central UI and reused when that dataset is selected.

If you prefer the older Toolkit-proxy mode for a full remote AITK Studio worker, run the UI with `AI_TOOLKIT_AUTH` and Cloudflared as above, and keep Ollama bound to localhost:

```bash
AITK_OLLAMA_ENABLED=1
AITK_OLLAMA_HOST=127.0.0.1:11434
AITK_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Docker and RunPod images include Ollama. When `AITK_OLLAMA_ENABLED=1`, the startup scripts launch `ollama serve` without exposing port `11434`; only the authenticated UI is exposed through Cloudflared. If the selected Ollama model is not installed, the remote endpoint pulls it automatically before captioning.

Threat model limit: direct Remote Ollama endpoints receive plaintext image and prompt payloads in their HTTP API. HTTPS or a protected tunnel protects transport; the older Toolkit-proxy mode also encrypts payloads at the application layer before the worker UI decrypts them for local Ollama. Neither mode protects against a compromised remote machine or Ollama process.

## RunPod

This fork includes a maintained private RunPod Pod template for the AITK Studio UI. See [`runpod/README.md`](../runpod/README.md) for the Blackwell-first template, persistent volume layout, required `AI_TOOLKIT_AUTH` secret, and access URL format: `https://<POD_ID>-8675.proxy.runpod.net`.

## Modal

### 1. Setup

#### AITK Studio

```bash
git clone https://github.com/BadAtCaptchas/AITK-Studio.git
cd AITK-Studio
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
python scripts/install_runtime.py
python scripts/environment_doctor.py
```

#### Modal
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate. If that does not work, try `python -m modal setup`.

#### Hugging Face
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, .png, .webp, or experimental .jxl images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config from `config/examples/modal` to the `config` folder and rename it to `whatever_you_want.yml`.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:

   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function`. The default is A100 40GB with a 2-hour timeout.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`.
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

## Upgrading from workspace releases

Stop the application and its workers before upgrading. Startup retains the existing SQLite backup mechanism and checks for obsolete records before modifying the database. Empty legacy structures are removed and global run-name uniqueness is restored. Nonempty legacy workspace records, associated jobs, saved scoped watchers, or conflicting run names stop the upgrade and require manual migration. The upgrade never moves or deletes dataset, model, output, or former workspace folders. Keep those folders and backups until you have reviewed any manual migration.

Obsolete scoped API requests are rejected explicitly; removed workspace routes are unavailable. Older workers' scoped jobs are excluded from global run discovery.
