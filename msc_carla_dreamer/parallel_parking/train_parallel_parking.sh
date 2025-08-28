#!/usr/bin/env bash
set -e
PORT=${1:-2000}
GPU=${2:-0}
EXTRA_OPTS=${@:3}


# Auto-detect Windows host IP when running from WSL2 if CARLA_HOST unset
if [ -z "${CARLA_HOST}" ]; then
    if grep -qi microsoft /proc/version; then
    # Prefer LAN IP if set, else nameserver gateway
        # HOST_IP=$(ip route get 1.1.1.1 | awk '{print $7; exit}')
        # export CARLA_HOST=${HOST_IP}
        export CARLA_HOST=10.140.13.244
    else
        export CARLA_HOST=10.140.13.244
    fi
fi
export CARLA_PORT=${PORT}


echo "Using CARLA at ${CARLA_HOST}:${CARLA_PORT} (GPU ${GPU})"


# Start training (arg style 1: argparse)
python -u dreamerv3/train.py \
    --task parallel_parking \
    --env.world.carla_host ${CARLA_HOST} \
    --env.world.carla_port ${CARLA_PORT} \
    --dreamerv3.jax.policy_devices ${GPU} \
    --dreamerv3.jax.train_devices ${GPU} \
    ${EXTRA_OPTS}


# If your repo uses Hydra-style configs, instead use:
# python -u dreamerv3/train.py --configs tasks/parallel_parking \
# env.world.carla_host=${CARLA_HOST} env.world.carla_port=${CARLA_PORT} \
# dreamerv3.jax.policy_devices=${GPU} dreamerv3.jax.train_devices=${GPU} \
# ${EXTRA_OPTS}