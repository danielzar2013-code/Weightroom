#!/usr/bin/env bash
# Weightroom GPU for Linux and Kaggle.
# Installs ComfyUI with the Weightroom bridge, Chatterbox speech, Ollama for chat and a Cloudflare tunnel,
# starts them, and prints a link (and a QR code) that connects the Weightroom app to this GPU.
# Run it again any time: it skips what is already installed.
set -euo pipefail

APP="https://danielzar2013-code.github.io/Weightroom/"
APP_ORIGIN="https://danielzar2013-code.github.io"
COMFY_TAG="v0.45.5"
CHATTERBOX_REV="f7d7a16187430abcaf91a3039b9c83aa9960816a"
if [ -d /kaggle ]; then DEFAULT_ROOT=/tmp/weightroom; else DEFAULT_ROOT="$HOME/weightroom-gpu"; fi
ROOT="${WR_ROOT:-$DEFAULT_ROOT}"
COMFY="$ROOT/ComfyUI"
LOGS="$ROOT/logs"
PORT="${WR_PORT:-8188}"
mkdir -p "$ROOT" "$LOGS"

say() { printf '\n== %s\n' "$*"; }
fetch() { curl -fL --retry 3 --retry-delay 2 -o "$2" "$1"; }

# ---------- GPU ----------
GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
if [ "$GPUS" -eq 0 ]; then
  echo "No NVIDIA GPU found. On Kaggle: Settings > Accelerator > GPU T4 x2, and turn Internet on." >&2
  exit 1
fi

# ---------- Python ----------
# Kaggle and Colab already ship PyTorch with CUDA; elsewhere make a venv with PyTorch for CUDA 12.8.
if python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  PY=python3
  PIP=(python3 -m pip install -q)
else
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    say "Installing uv (Python manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  PY="$ROOT/venv/bin/python"
  if [ ! -x "$PY" ]; then
    say "Setting up Python 3.12 and PyTorch (about 3 GB, one time)"
    "$UV" venv -q --python 3.12 "$ROOT/venv"
    "$UV" pip install -q --python "$PY" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  fi
  PIP=("$UV" pip install -q --python "$PY")
fi
# Keep the CUDA build of PyTorch: other packages may not replace it.
"$PY" -c 'import torch; print("torch==" + torch.__version__)' > "$ROOT/constraints.txt"

# ---------- ComfyUI ----------
if [ ! -f "$COMFY/main.py" ]; then
  say "Downloading ComfyUI $COMFY_TAG"
  git clone -q --depth 1 --branch "$COMFY_TAG" https://github.com/Comfy-Org/ComfyUI "$COMFY"
fi
if [ "$(cat "$ROOT/.deps" 2>/dev/null || true)" != "$COMFY_TAG" ]; then
  say "Installing ComfyUI requirements"
  "${PIP[@]}" -r "$COMFY/requirements.txt" qrcode -c "$ROOT/constraints.txt"
  echo "$COMFY_TAG" > "$ROOT/.deps"
fi

# ---------- add-ons ----------
mkdir -p "$COMFY/custom_nodes/weightroom_bridge"
fetch "${APP}gpu/weightroom_bridge/__init__.py" "$COMFY/custom_nodes/weightroom_bridge/__init__.py"
CB="$COMFY/custom_nodes/ComfyUI_Fill-ChatterBox"
if [ ! -f "$CB/.weightroom-ok" ]; then
  say "Adding Chatterbox speech (Hebrew and 22 more languages)"
  rm -rf "$CB"
  git clone -q https://github.com/filliptm/ComfyUI_Fill-ChatterBox "$CB"
  git -C "$CB" checkout -q "$CHATTERBOX_REV"
  "${PIP[@]}" -r "$CB/requirements.txt" -c "$ROOT/constraints.txt"
  touch "$CB/.weightroom-ok"
fi

# ---------- Ollama (chat) ----------
if ! command -v ollama >/dev/null 2>&1; then
  say "Installing Ollama for chat"
  if [ "$(id -u)" -eq 0 ]; then
    # Ollama's installer needs zstd to unpack, and Kaggle doesn't have it.
    if ! command -v zstd >/dev/null 2>&1; then
      if command -v apt-get >/dev/null 2>&1; then
        (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd) >/dev/null 2>&1 || true
      elif command -v dnf >/dev/null 2>&1; then
        dnf install -y -q zstd >/dev/null 2>&1 || true
      fi
    fi
    curl -fsSL https://ollama.com/install.sh | sh || echo "Ollama didn't install. Chat stays off; everything else works." >&2
  else
    echo "Ollama needs root to install here. Chat stays off; everything else works." >&2
  fi
fi

# ---------- tunnel ----------
CF="$ROOT/cloudflared"
if [ ! -x "$CF" ]; then
  fetch https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "$CF"
  chmod +x "$CF"
fi

# ---------- key ----------
KEYFILE="$ROOT/key.txt"
if [ ! -s "$KEYFILE" ]; then
  "$PY" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$KEYFILE"
fi
KEY="$(tr -d '\r\n ' < "$KEYFILE")"

# ---------- start ----------
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

COMFY_GPU=""; OLLAMA_GPU=""
if [ "$GPUS" -ge 2 ]; then COMFY_GPU=0; OLLAMA_GPU=1; SHARED=0; else SHARED=1; fi

if command -v ollama >/dev/null 2>&1 && ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if [ -n "$OLLAMA_GPU" ]; then
    CUDA_VISIBLE_DEVICES="$OLLAMA_GPU" OLLAMA_KEEP_ALIVE=10m nohup ollama serve > "$LOGS/ollama.log" 2>&1 &
  else
    OLLAMA_KEEP_ALIVE=5m nohup ollama serve > "$LOGS/ollama.log" 2>&1 &
  fi
  PIDS+=($!)
fi

say "Starting ComfyUI (the first start takes a minute or two)"
(
  cd "$COMFY"
  if [ -n "$COMFY_GPU" ]; then export CUDA_VISIBLE_DEVICES="$COMFY_GPU"; fi
  WEIGHTROOM_KEY="$KEY" WEIGHTROOM_SHARED_GPU="$SHARED" exec "$PY" main.py --listen 127.0.0.1 --port "$PORT" --enable-cors-header "$APP_ORIGIN"
) > "$LOGS/comfyui.log" 2>&1 &
COMFY_PID=$!
PIDS+=("$COMFY_PID")
for _ in $(seq 1 300); do
  if curl -fs "http://127.0.0.1:$PORT/weightroom/info" -H "Authorization: Bearer $KEY" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then echo "ComfyUI stopped. Last lines of its log:" >&2; tail -n 40 "$LOGS/comfyui.log" >&2; exit 1; fi
  sleep 2
done

say "Opening a secure tunnel"
"$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$LOGS/tunnel.log" 2>&1 &
PIDS+=($!)
URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -n1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
if [ -z "$URL" ]; then echo "The tunnel did not start. Its log:" >&2; tail -n 30 "$LOGS/tunnel.log" >&2; exit 1; fi

LINK="$("$PY" - "$URL" "$KEY" "$APP" <<'PYCODE'
import base64, json, sys
url, key, app = sys.argv[1:4]
blob = base64.urlsafe_b64encode(json.dumps({"u": url, "k": key}).encode()).decode().rstrip("=")
print(app + "#gpu=" + blob)
PYCODE
)"

say "Your GPU is ready"
echo "Open this link on your phone or computer to connect the Weightroom app:"
echo
echo "$LINK"
echo
"$PY" - "$LINK" <<'PYCODE' || true
import sys, qrcode
q = qrcode.QRCode(border=1)
q.add_data(sys.argv[1])
q.print_ascii(invert=True)
PYCODE
echo
echo "Keep this running while you use your GPU. Stop it to disconnect. Logs: $LOGS"
echo "The link changes each time you start. Your key stays the same (in $KEYFILE)."
wait "$COMFY_PID"
