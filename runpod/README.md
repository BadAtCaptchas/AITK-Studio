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
- UI loads but jobs cannot download gated models: add your Hugging Face token in the UI settings or provide `HF_TOKEN` as a RunPod secret-backed environment variable.
- TensorBoard link is missing: set `AITK_ENABLE_TENSORBOARD=1` and restart the Pod.
- Data disappeared after restart: confirm the Pod has a network volume mounted at `/workspace`; data outside `/workspace` is ephemeral.
