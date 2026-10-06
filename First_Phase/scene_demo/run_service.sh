#!/usr/bin/env bash
# Launcher for the persistent-scene SmolVLA-on-LIBERO HTTP service (v2).
#
# Pure launcher: it reuses the exact GL / LIBERO / offline configuration of the
# v1 libero_demo launcher, then execs scene_demo/service.py on the fixed default
# port 8767 with the persistent scene run root. It installs nothing and
# downloads nothing.
set -euo pipefail

# Headless EGL rendering for MuJoCo / LIBERO.
export MUJOCO_GL=egl

# LIBERO reads its config.yaml from this directory (written by setup_env.py).
export LIBERO_CONFIG_PATH=/home/yhwang/fyp/libero_demo/libero_config

# WSL exposes the host CUDA/GL drivers here.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

# Never reach the network for the checkpoint, base VLM or tokenizer.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Drop any inherited proxy configuration.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY || true

exec /home/yhwang/fyp/libero_demo/venv/bin/python -u \
    /mnt/d/FYP/First_Phase/scene_demo/service.py \
    --port 8767 \
    --run-root /home/yhwang/fyp/scene_demo/runs \
    --completion-mode release_verified "$@"
