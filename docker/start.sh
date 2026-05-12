#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ref https://github.com/runpod/containers/blob/main/container-template/start.sh

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #


# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh

         # Regenerate SSH host keys on every container startup so image-baked keys are never reused.
        rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub
        ssh-keygen -A

        service ssh start

        echo "SSH host keys:"
        for key in /etc/ssh/*.pub; do
            echo "Key: $key"
            ssh-keygen -lf $key
        done
    fi
}


require_auth_secret() {
    if [[ -z "${AI_TOOLKIT_AUTH:-}" ]]; then
        echo "ERROR: AI_TOOLKIT_AUTH is required and must be a strong bearer token." >&2
        exit 1
    fi

    if [[ "${AI_TOOLKIT_AUTH}" == "password" ]]; then
        echo "ERROR: AI_TOOLKIT_AUTH must not use the insecure default value: password" >&2
        exit 1
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #


echo "Pod Started"

setup_ssh
export_env_vars
require_auth_secret
echo "Starting OstrisAI-Toolkit Revamped UI..."
cd /app/ai-toolkit/ui && npm run update_db && npm run start
