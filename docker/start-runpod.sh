#!/bin/bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app/ai-toolkit}"
WORKSPACE_ROOT="${RUNPOD_WORKSPACE:-/workspace}"
PERSIST_ROOT="${AITK_PERSIST_ROOT:-${WORKSPACE_ROOT}/ai-toolkit}"

export AITK_PERSIST_ROOT="${PERSIST_ROOT}"
export AITK_DB_PROVIDER="${AITK_DB_PROVIDER:-sqlite}"
export AITK_SQLITE_PATH="${AITK_SQLITE_PATH:-${PERSIST_ROOT}/aitk_db.db}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export AITK_TENSORBOARD_PORT="${AITK_TENSORBOARD_PORT:-6006}"
export AITK_TENSORBOARD_HOST="${AITK_TENSORBOARD_HOST:-0.0.0.0}"

truthy() {
    case "${1,,}" in
        1|true|yes|on|enabled) return 0 ;;
        *) return 1 ;;
    esac
}

setup_ssh() {
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        return
    fi

    echo "Setting up SSH..."
    mkdir -p ~/.ssh /var/run/sshd
    echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub
    ssh-keygen -A
    service ssh start

    echo "SSH host keys:"
    for key in /etc/ssh/*.pub; do
        echo "Key: ${key}"
        ssh-keygen -lf "${key}"
    done
}

write_runpod_env() {
    echo "Exporting RunPod environment variables..."
    : > /etc/rp_environment
    while IFS='=' read -r key value; do
        case "${key}" in
            RUNPOD_*|PATH|_)
                printf 'export %s=%q\n' "${key}" "${value}" >> /etc/rp_environment
                ;;
        esac
    done < <(printenv)

    if ! grep -q 'source /etc/rp_environment' ~/.bashrc 2>/dev/null; then
        echo 'source /etc/rp_environment' >> ~/.bashrc
    fi
}

require_auth_secret() {
    if [[ -z "${AI_TOOLKIT_AUTH:-}" ]]; then
        echo "ERROR: AI_TOOLKIT_AUTH is required and must be a strong bearer token." >&2
        exit 1
    fi

    case "${AI_TOOLKIT_AUTH}" in
        password|change_me|changeme|"{{ RUNPOD_SECRET_ai_toolkit_auth }}")
            echo "ERROR: AI_TOOLKIT_AUTH must be set to a real secret value." >&2
            exit 1
            ;;
    esac
}

validate_cloudflared_config() {
    if [[ "${AITK_CLOUDFLARED_ENABLED:-}" != "1" ]]; then
        return
    fi

    require_auth_secret

    if ! command -v "${AITK_CLOUDFLARED_BIN:-cloudflared}" >/dev/null 2>&1; then
        echo "ERROR: cloudflared is enabled but the cloudflared binary was not found." >&2
        exit 1
    fi

    if [[ -z "${AITK_CLOUDFLARED_PUBLIC_URL:-}" ]]; then
        echo "ERROR: AITK_CLOUDFLARED_PUBLIC_URL is required when cloudflared is enabled." >&2
        exit 1
    fi

    if [[ -z "${AITK_CLOUDFLARED_TOKEN_FILE:-}" || ! -f "${AITK_CLOUDFLARED_TOKEN_FILE}" ]]; then
        echo "ERROR: AITK_CLOUDFLARED_TOKEN_FILE must point to an existing tunnel token file." >&2
        exit 1
    fi
}

start_ollama() {
    if [[ "${AITK_OLLAMA_ENABLED:-0}" != "1" ]]; then
        return
    fi

    if ! command -v ollama >/dev/null 2>&1; then
        echo "ERROR: AITK_OLLAMA_ENABLED=1 but the ollama binary was not found." >&2
        exit 1
    fi

    export OLLAMA_HOST="${AITK_OLLAMA_HOST:-127.0.0.1:11434}"
    export AITK_OLLAMA_BASE_URL="${AITK_OLLAMA_BASE_URL:-http://${OLLAMA_HOST}}"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-${PERSIST_ROOT}/ollama}"
    mkdir -p "${OLLAMA_MODELS}"
    echo "Starting Ollama on ${OLLAMA_HOST} with models at ${OLLAMA_MODELS}..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
}

link_persistent_dir() {
    local name="$1"
    local target="${PERSIST_ROOT}/${name}"
    local link="${APP_ROOT}/${name}"

    mkdir -p "${target}"

    if [[ -L "${link}" ]]; then
        if [[ "$(readlink "${link}")" != "${target}" ]]; then
            rm -f "${link}"
            ln -s "${target}" "${link}"
        fi
        return
    fi

    if [[ -e "${link}" ]]; then
        if [[ -d "${link}" && -z "$(find "${link}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            rmdir "${link}"
        else
            local backup="${link}.image"
            if [[ ! -e "${backup}" ]]; then
                mv "${link}" "${backup}"
            else
                rm -rf "${link}"
            fi
        fi
    fi

    ln -s "${target}" "${link}"
}

prepare_workspace() {
    mkdir -p \
        "${PERSIST_ROOT}/datasets" \
        "${PERSIST_ROOT}/output" \
        "${PERSIST_ROOT}/config" \
        "${PERSIST_ROOT}/models" \
        "${PERSIST_ROOT}/data" \
        "${HF_HOME}" \
        "${HUGGINGFACE_HUB_CACHE}" \
        "$(dirname "${AITK_SQLITE_PATH}")"

    if [[ ! -e "${PERSIST_ROOT}/config/examples" && -d "${APP_ROOT}/config/examples" ]]; then
        echo "Copying example configs to persistent storage..."
        cp -a "${APP_ROOT}/config/examples" "${PERSIST_ROOT}/config/"
    fi

    link_persistent_dir datasets
    link_persistent_dir output
    link_persistent_dir config
    link_persistent_dir models
    link_persistent_dir data
}

configure_runpod_proxy_urls() {
    if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
        return
    fi

    if [[ -z "${AITK_TENSORBOARD_PUBLIC_URL:-}" ]] && truthy "${AITK_ENABLE_TENSORBOARD:-0}"; then
        export AITK_TENSORBOARD_PUBLIC_URL="https://${RUNPOD_POD_ID}-${AITK_TENSORBOARD_PORT}.proxy.runpod.net"
    fi
}

echo "RunPod AI Toolkit container starting..."

setup_ssh
write_runpod_env
require_auth_secret
validate_cloudflared_config
prepare_workspace
configure_runpod_proxy_urls
start_ollama

echo "Persistent root: ${PERSIST_ROOT}"
echo "SQLite path: ${AITK_SQLITE_PATH}"
echo "Hugging Face cache: ${HUGGINGFACE_HUB_CACHE}"
echo "Starting OstrisAI-Toolkit Revamped UI on port 8675..."

cd "${APP_ROOT}/ui"
npm run update_db
exec npm run start
