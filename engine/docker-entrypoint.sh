#!/bin/sh
# Pick a MuJoCo GL backend unless the caller forced one: probe EGL when a GPU
# device node is exposed, falling back to software OSMesa if EGL cannot create
# an offscreen context (e.g. non-NVIDIA DRI nodes or unprivileged containers).
NVIDIA_DEV="${RC_NVIDIA_DEV:-/dev/nvidiactl}"
DRI_DEV="${RC_DRI_DEV:-/dev/dri/renderD128}"

if [ -z "$MUJOCO_GL" ]; then
    if [ -e "$NVIDIA_DEV" ] || [ -e "$DRI_DEV" ]; then
        if MUJOCO_GL=egl python -c 'import mujoco; ctx = mujoco.GLContext(64, 48); ctx.free()' >/dev/null 2>&1; then
            export MUJOCO_GL=egl
        else
            echo "[rc_engine] EGL probe failed; falling back to MUJOCO_GL=osmesa" >&2
            export MUJOCO_GL=osmesa
        fi
    else
        export MUJOCO_GL=osmesa
    fi
fi
echo "[rc_engine] MUJOCO_GL=$MUJOCO_GL" >&2
exec python -m rc_engine "$@"
