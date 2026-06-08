# RunPod Blackwell Template

This folder contains the private RunPod Pod template for OstrisAI-Toolkit Revamped. It is intended for NVIDIA Blackwell-first deployments with a reproducible Docker image and persistent data on the RunPod volume at `/workspace`.

## Build And Push

Build the image from the repository root:

```bash
IMAGE_NAME="REPLACE_WITH_YOUR_REGISTRY/ai-toolkit:0.9.10-blackwell"

docker build \
  -f docker/Dockerfile.runpod \
  --build-arg AITK_VERSION=0.9.10 \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
  -t "${IMAGE_NAME}" \
  .
```

Push the image to your registry:

```bash
docker push "${IMAGE_NAME}"
```

Before creating the RunPod template, replace `REPLACE_WITH_YOUR_REGISTRY/ai-toolkit:0.9.10-blackwell` in `runpod/template.blackwell.json` with the exact image name you pushed.

## Create The Template

Create a RunPod secret named `ai_toolkit_auth` with a strong bearer token. The template references it as:

```txt
{{ RUNPOD_SECRET_ai_toolkit_auth }}
```

If you want durable resume for encrypted datasets, also create `aitk_durable_dataset_key_secret` with a separate random value of at least 32 characters. Without it, encrypted jobs can still start with a supplied key, but durable encrypted resume is rejected. YubiKey-protected datasets are unlocked in the browser attached to the central UI; the RunPod worker receives only the ephemeral dataset key during the authenticated HTTPS job start and does not need USB access.

In the RunPod console, create a private Pod template with the values from `template.blackwell.json`:

- Image: the pushed value of `IMAGE_NAME`
- Container disk: `100 GB`
- Volume: `250 GB`
- Volume mount path: `/workspace`
- HTTP ports: `8675`, `6006`
- TCP ports: `22`
- Environment variables: copy the `env` object from `template.blackwell.json`

You can also create it with the REST API:

```bash
curl --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data @runpod/template.blackwell.json
```

## Runtime Paths

On startup, `docker/start-runpod.sh` creates persistent storage under `/workspace/ai-toolkit` and links these paths back into `/app/ai-toolkit`:

- `datasets`
- `output`
- `config`
- `models`
- `data`

The SQLite database lives at `/workspace/ai-toolkit/aitk_db.db`, and Hugging Face files cache under `/workspace/.cache/huggingface`. Example configs are copied to `/workspace/ai-toolkit/config/examples` on first boot.

## Access

After the Pod starts, open the UI through RunPod's HTTP proxy:

```txt
https://<POD_ID>-8675.proxy.runpod.net
```

Log in with the value stored in the `ai_toolkit_auth` RunPod secret. The container refuses to start when `AI_TOOLKIT_AUTH` is missing or still set to a placeholder value.

SSH starts only when RunPod provides `PUBLIC_KEY`. Use the SSH connection details shown in the RunPod Connect panel.

## TensorBoard

TensorBoard is disabled by default:

```txt
AITK_ENABLE_TENSORBOARD=0
```

To enable it, edit the Pod or template and set:

```txt
AITK_ENABLE_TENSORBOARD=1
AITK_TENSORBOARD_HOST=0.0.0.0
AITK_TENSORBOARD_PORT=6006
```

When running on RunPod, the startup script sets `AITK_TENSORBOARD_PUBLIC_URL` to:

```txt
https://<POD_ID>-6006.proxy.runpod.net
```

TensorBoard is not protected by `AI_TOOLKIT_AUTH`; expose it only when you need it.

## Secure Remote Ollama Captioning

The image includes Ollama for secure remote caption workers, but it is disabled by default:

```txt
AITK_OLLAMA_ENABLED=0
AITK_OLLAMA_HOST=127.0.0.1:11434
AITK_OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODELS=/workspace/ai-toolkit/ollama
```

To use this Pod as a standalone Remote Ollama endpoint, set `AITK_OLLAMA_ENABLED=1` and expose Ollama through a protected HTTP(S) route or trusted private network. Add that Ollama base URL in the central UI under **Settings > Remote Ollama**. The Pod does not need to run the AI Toolkit UI for direct Remote Ollama captioning.

For the older full-worker Toolkit proxy mode, keep `OLLAMA_HOST` on `127.0.0.1:11434` and expose only the AI Toolkit UI through RunPod or Cloudflared. The central UI sends encrypted per-image payloads, including dataset-specific system prompts when configured, to the worker UI; datasets are not uploaded or stored on the worker. Missing Ollama models are pulled automatically.

## Smoke Checks

Inside the Pod terminal:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
test -f /workspace/ai-toolkit/aitk_db.db
test -d /workspace/.cache/huggingface/hub
```

Restart the Pod and confirm that jobs, datasets, outputs, and the SQLite database are still present under `/workspace/ai-toolkit`.

## Troubleshooting

- UI exits immediately: confirm the `ai_toolkit_auth` secret exists and is not empty.
- Durable encrypted resume is rejected: confirm `AITK_DURABLE_DATASET_KEY_SECRET` is set from the `aitk_durable_dataset_key_secret` RunPod secret and was not changed after queuing the job.
- UI loads but jobs cannot download gated models: add your Hugging Face token in the UI settings or provide `HF_TOKEN` as a RunPod secret-backed environment variable.
- TensorBoard link is missing: set `AITK_ENABLE_TENSORBOARD=1` and restart the Pod.
- Data disappeared after restart: confirm the Pod has a network volume mounted at `/workspace`; data outside `/workspace` is ephemeral.
