# syntax=docker/dockerfile:1
# RoboDojo simulator / evaluation image (Ubuntu 22.04 + CUDA 12.8).
#
# Scope: this image contains ONLY the RoboDojo simulation-evaluation side
# (Isaac Sim, IsaacLab, CuRobo, the RoboDojo Python stack) plus the lightweight
# XPolicyLab client_server + policy deploy adapters used by robodojo.sim.evaluation.
# It deliberately does NOT install any policy-specific environment, dependency
# set, or checkpoints (e.g. GR00T_N17). The policy server runs OUTSIDE this
# container and is reached over TCP (see docker/README.md).
#
# Build:
#   docker build -t robodojo:cuda12.8 .
#
# Run (Linux host, policy server already listening on the host):
#   docker run --rm -it --gpus all --network host --ipc host \
#     -v "$PWD/.robodojo:/workspace/RoboDojo/.robodojo" \
#     robodojo:cuda12.8 \
#     client --task stack_bowls \
#       --policy-name GR00T_N17 --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDA_HOME=/usr/local/cuda \
    PATH=/workspace/RoboDojo/.venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    TERM=xterm-256color \
    PYTHONNOUSERSITE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/RoboDojo/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.post1.dev100 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# ── Optional China mirrors (build args; defaults = official upstreams) ─────────
# The image is unchanged for normal builds. On a China network, enable mirrors:
#   docker build \
#     --build-arg UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#     --build-arg PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     -t robodojo:cuda12.8 .
# (robodojo docker smoke sets all of these when ROBODOJO_CN_MIRRORS=1.)
ARG UBUNTU_MIRROR=""
ARG PYPI_INDEX_URL="https://pypi.org/simple"
ENV UV_DEFAULT_INDEX=${PYPI_INDEX_URL}

# ── System dependencies ──────────────────────────────────────────────────────
# RoboDojo machine prerequisites plus
# the headless OpenGL/EGL/Vulkan runtime libraries Isaac Sim needs, and
# netcat for the policy-server connectivity check documented in docker/README.md.
RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
        sed -i "s@http://archive.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g; s@http://security.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g" /etc/apt/sources.list 2>/dev/null || true; \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        wget \
        curl \
        ca-certificates \
        awscli \
        ffmpeg \
        netcat-openbsd \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libegl1 \
        libvulkan1 \
        vulkan-tools \
        mesa-utils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Headless EGL / Vulkan ICD (NVIDIA) ───────────────────────────────────────
# Point the GL/Vulkan loaders at the NVIDIA driver that the NVIDIA Container
# Toolkit injects at runtime, so Isaac Sim can render cameras headless.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.242"
    }
}
EOF

WORKDIR /workspace/RoboDojo

# ── Locked uv environment ─────────────────────────────────────────────────────
# Copy dependency metadata, the root package, and the cuRobo compatibility fork
# before application code so the simulator dependency layer remains cached.
COPY pyproject.toml uv.lock README.md LICENSE /workspace/RoboDojo/
COPY src/ /workspace/RoboDojo/src/
COPY third_party/curobo/ /workspace/RoboDojo/third_party/curobo/
# Submodule .git files point outside the Docker context. Remove them before uv
# asks setuptools-scm to build the local editable cuRobo package.
RUN find /workspace/RoboDojo/third_party/curobo -name .git -prune -exec rm -rf {} + 2>/dev/null; true
RUN uv python install 3.11 && uv sync --extra sim --locked --no-dev --no-cache

# ── RoboDojo source + XPolicyLab client/adapter code ─────────────────────────
# Explicit copies keep heavy policy source, weights, checkpoints, assets, and
# runtime outputs out of the image via .dockerignore.
COPY configs/ /workspace/RoboDojo/configs/
COPY scripts/eval_policy.sh /workspace/RoboDojo/scripts/eval_policy.sh
COPY XPolicyLab/ /workspace/RoboDojo/XPolicyLab/

# ── Headless RTX render deps + single Vulkan ICD (late, cheap layers) ─────────
SHELL ["/bin/bash", "-c"]
# libXt.so.6 is required by Isaac Sim's MaterialX render libs
# (libusd_usdBakeMtlx.so, libMaterialXRender*.so). Without it they fail to load at
# startup ("libXt.so.6: cannot open shared object file"). Its deps libsm6/libice6
# are already installed above. Kept as a late layer so the expensive
# simulator dependency layers stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends libxt6 \
    && rm -rf /var/lib/apt/lists/*
# Force a SINGLE Vulkan ICD. This image bakes /usr/share/vulkan/icd.d/nvidia_icd.json
# AND the NVIDIA Container Toolkit injects /etc/vulkan/icd.d/nvidia_icd.json at run
# time; the loader then sees two ICDs for the same GPU ("multiple installable client
# drivers") and RTX device init wedges on the first rendered frame (tiled cameras
# hang right after buffer allocation). Pinning the toolkit-injected, driver-matched
# ICD makes headless RTX rendering start reliably. --gpus with
# NVIDIA_DRIVER_CAPABILITIES=all guarantees that path exists at run time.
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# ── Entrypoint ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace/RoboDojo/XPolicyLab

ENTRYPOINT ["/workspace/RoboDojo/.venv/bin/robodojo"]
CMD ["--help"]
