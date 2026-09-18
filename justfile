# gliner2-doom — see AGENTS.md. No system deps: everything below lands in this dir or in caches.

ort_dir := justfile_directory() + "/vendor/onnxruntime"
export ORT_DYLIB_PATH := ort_dir + "/lib/libonnxruntime.dylib"
export GLINER2_DOOM_PYTHON := env_var_or_default("GLINER2_DOOM_PYTHON", env_var("HOME") + "/virtualenvs/gliner2-doom-vizdoom/bin/python")

default:
    @just --list

# Download ONNX Runtime 1.28.2 (macOS arm64) into vendor/. Matches ort 2.0.0-rc.13.
fetch-ort:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f "{{ort_dir}}/lib/libonnxruntime.dylib" ] && { echo "ort present"; exit 0; }
    mkdir -p vendor && cd vendor
    curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.28.2/onnxruntime-osx-arm64-1.28.2.tgz
    tar xzf ort.tgz && rm ort.tgz && mv onnxruntime-osx-arm64-1.28.2 onnxruntime
    rm -rf onnxruntime/lib/*.dSYM onnxruntime/lib/cmake onnxruntime/lib/pkgconfig

# Create the ViZDoom venv (uv, py3.12) under ~/virtualenvs.
setup-venv:
    uv venv ~/virtualenvs/gliner2-doom-vizdoom --python 3.12
    uv pip install --python ~/virtualenvs/gliner2-doom-vizdoom/bin/python vizdoom==1.3.0

setup: fetch-ort setup-venv
    cargo build --release

# Latency + RSS benchmark of classification-only decisions (S1 gate).
bench *ARGS:
    cargo run --release -- bench {{ARGS}}

# Real-time Doom in the terminal (q quit, p pause, space step, m manual).
live *ARGS:
    cargo run --release -- live {{ARGS}}

# Synchronous play: the engine waits for each decision. --record x.jsonl for per-decision logs.
play *ARGS:
    cargo run --release -- play {{ARGS}}

# Business twin on built-in support tickets (or --file requests.jsonl).
triage *ARGS:
    cargo run --release -- triage {{ARGS}}

# System One endpoint on :8000.
serve *ARGS:
    cargo run --release -- serve {{ARGS}}

# Score situations against label sets while designing prompts.
probe *ARGS:
    cargo run --release -- probe {{ARGS}}

# Direct policy, step 1: oracle + epsilon-random rollouts -> data/{train,val,test}.jsonl (appends).
collect *ARGS:
    cargo run --release -- collect {{ARGS}}

# Direct policy, step 2: train the option-attention head (candle, Metal by default). Encoder states cached in data/states.metal.bin.
train *ARGS:
    cargo run --release -- train {{ARGS}}

# Direct policy, step 3: offline metrics on a split.
eval *ARGS:
    cargo run --release -- eval {{ARGS}}
