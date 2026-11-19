"""Weightroom GPU bridge.

A ComfyUI add-on that lets the Weightroom app (https://danielzar2013-code.github.io/Weightroom/)
use this ComfyUI from a phone or another computer:

* every request must carry this machine's Weightroom key, so a public tunnel link alone is useless;
* /weightroom/info reports the GPU and which Weightroom models are installed;
* /weightroom/install downloads a model's files from Hugging Face into the ComfyUI model folders
  (only the files listed in CATALOG below, never an address sent by the browser);
* /weightroom/chat forwards chat messages to Ollama running on the same machine.
"""
import asyncio
import hmac
import ipaddress
import json
import logging
import os
import secrets
import threading
import time
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

import folder_paths
from server import PromptServer

VERSION = 1
log = logging.getLogger("weightroom")
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434").rstrip("/")
# One GPU for both ComfyUI and Ollama (a home PC): each frees the GPU for the other. Kaggle gives them a card each.
SHARED_GPU = os.environ.get("WEIGHTROOM_SHARED_GPU", "1") != "0"
HF = "https://huggingface.co/"

# Weightroom model id -> what it needs. File entries: (ComfyUI model folder, file name, Hugging Face path).
CATALOG = {
    "minimax-h3-singularity": {"name": "MiniMax H3 Singularity V2", "files": [
        ("diffusion_models", "Minimax-h3_Singularity_ref2va_v2_pruned_w4a8.safetensors", "WAS/Minimax-h3_Singularity_v2_pruned/resolve/main/Minimax-h3_Singularity_ref2va_v2_pruned_w4a8.safetensors"),
        ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
        ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors"),
        ("vae", "minimax_h3_audio_vae_fp32.safetensors", "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"),
        ("loras", "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", "lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"),
        ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", "LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ]},
    "z-image-turbo": {"name": "Z-Image Turbo", "files": [
        ("diffusion_models", "z_image_turbo_bf16.safetensors", "Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
        ("text_encoders", "qwen_3_4b.safetensors", "Comfy-Org/Z-Image-Turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
        ("vae", "ae.safetensors", "Comfy-Org/Z-Image-Turbo/resolve/main/split_files/vae/ae.safetensors"),
    ]},
    "wan22-5b": {"name": "Wan 2.2 5B", "files": [
        ("diffusion_models", "wan2.2_ti2v_5B_fp16.safetensors", "Comfy-Org/Wan-2.2/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
        ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        ("vae", "wan2.2_vae.safetensors", "Comfy-Org/Wan-2.2/resolve/main/split_files/vae/wan2.2_vae.safetensors"),
    ]},
    "hunyuan3d-21": {"name": "Hunyuan3D 2.1", "files": [
        ("checkpoints", "hunyuan_3d_v2.1.safetensors", "Comfy-Org/Hunyuan-3D-2.1/resolve/main/hunyuan_3d_v2.1.safetensors"),
    ]},
    "ace-step-15": {"name": "ACE-Step 1.5", "files": [
        ("checkpoints", "ace_step_1.5_turbo_aio.safetensors", "Comfy-Org/ACE-Step-1.5/resolve/main/checkpoints/ace_step_1.5_turbo_aio.safetensors"),
    ]},
    "chatterbox-ml": {"name": "Chatterbox Multilingual", "node": "FL_ChatterboxMultilingualTTS", "files": [
        ("chatterbox/chatterbox_multilingual", name, "ResembleAI/chatterbox/resolve/main/" + name)
        for name in ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]
    ]},
    "gpt-oss-20b": {"name": "gpt-oss-20b", "ollama": "gpt-oss:20b"},
}

# ---------- the key ----------

def _load_key():
    key = os.environ.get("WEIGHTROOM_KEY", "").strip()
    if key:
        return key
    path = os.path.join(folder_paths.get_user_directory(), "weightroom_key.txt")
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
    except OSError:
        key = ""
    if not key:
        key = secrets.token_urlsafe(24)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(key)
    return key


KEY = _load_key()


def _same(a, b):
    return bool(a) and hmac.compare_digest(a.encode(), b.encode())


def _is_loopback(host):
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _local_page(request):
    """ComfyUI's own page opened on this machine, not through the tunnel and not from another site."""
    h = request.headers
    if "Cf-Connecting-Ip" in h or "X-Forwarded-For" in h or h.get("Sec-Fetch-Site") == "cross-site":
        return False
    if not (_is_loopback(request.remote) and _is_loopback(urlparse("//" + (request.host or "")).hostname)):
        return False
    origin = h.get("Origin")
    return not origin or _is_loopback(urlparse(origin).hostname)


def allowed(request):
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and _same(auth[7:].strip(), KEY):
        return True
    if _same(request.query.get("wrkey", ""), KEY) or _same(request.cookies.get("wr_key", ""), KEY):
        return True
    return _local_page(request)


@web.middleware
async def key_check(request, handler):
    if request.method == "OPTIONS":
        return await handler(request)
    if not allowed(request):
        return web.json_response({"error": "This ComfyUI needs the Weightroom key. Open the connect link again."}, status=401)
    if SHARED_GPU and request.method == "POST" and request.path == "/prompt":
        await _unload_ollama()
    response = await handler(request)
    # Opening ComfyUI's own page with ?wrkey=... remembers the key for that page's later requests.
    if _same(request.query.get("wrkey", ""), KEY) and isinstance(response, web.StreamResponse) and not response.prepared:
        response.set_cookie("wr_key", KEY, max_age=30 * 86400, httponly=True, samesite="Lax", secure=request.secure or "Cf-Connecting-Ip" in request.headers)
    return response


# ---------- sharing one GPU ----------

async def _unload_ollama():
    """Ask Ollama to drop its loaded chat models so ComfyUI gets the whole GPU."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(OLLAMA + "/api/ps") as r:
                loaded = [m.get("name") for m in (await r.json(content_type=None)).get("models", [])]
            for name in loaded:
                async with s.post(OLLAMA + "/api/generate", json={"model": name, "keep_alive": 0}) as r:
                    await r.read()
    except Exception:
        pass


def _comfy_busy():
    running, pending = PromptServer.instance.prompt_queue.get_current_queue()
    return bool(running or pending)


def _free_comfy():
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()


# ---------- installs ----------

JOBS = {}  # model id -> {"state": "downloading"|"done"|"error", "done": bytes, "total": bytes, "error": str}
JOBS_LOCK = threading.Lock()


def _target(folder, name):
    if "/" in folder:  # an add-on's own folder under ComfyUI/models, like chatterbox/chatterbox_multilingual
        base = os.path.join(folder_paths.base_path, "models", *folder.split("/"))
    else:
        paths = folder_paths.get_folder_paths(folder)
        base = next((p for p in paths if os.path.basename(p) == folder), paths[0])
    return os.path.join(base, name)


def _have_file(folder, name):
    if "/" not in folder and folder_paths.get_full_path(folder, name):
        return True
    return os.path.isfile(_target(folder, name))


def _node_ready(entry):
    if not entry.get("node"):
        return True
    import nodes
    return entry["node"] in nodes.NODE_CLASS_MAPPINGS


async def _ollama_models():
    """Model tags Ollama has, or None when Ollama is not running."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
            async with s.get(OLLAMA + "/api/tags") as r:
                return [m.get("name", "") for m in (await r.json(content_type=None)).get("models", [])]
    except Exception:
        return None


def status(model_id, ollama_models, job):
    entry = CATALOG[model_id]
    if entry.get("ollama"):
        if ollama_models is None:
            return {"installed": False, "ready": False, "missing": "ollama", **job}
        tag = entry["ollama"]
        have = any(m == tag or m == tag + ":latest" for m in ollama_models)
        return {"installed": have, "ready": have, **job}
    have = all(_have_file(f, n) for f, n, _ in entry["files"])
    node = _node_ready(entry)
    out = {"installed": have, "ready": have and node, **job}
    if not node:
        out["missing"] = "addon"
    return out


def _download(model_id):
    import requests
    entry = CATALOG[model_id]
    todo = [(f, n, p) for f, n, p in entry["files"] if not _have_file(f, n)]
    try:
        sizes = []
        for f, n, p in todo:
            r = requests.head(HF + p, allow_redirects=True, timeout=30)
            sizes.append(int(r.headers.get("content-length") or 0))
        with JOBS_LOCK:
            JOBS[model_id] = {"state": "downloading", "done": 0, "total": sum(sizes)}
        finished = 0
        for (f, n, p), size in zip(todo, sizes):
            dest = _target(f, n)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            part = dest + ".part"
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(HF + p, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 200:
                    have = 0  # the server ignored the range; start over
                elif r.status_code != 206:
                    r.raise_for_status()
                with open(part, "ab" if have else "wb") as out:
                    for chunk in r.iter_content(chunk_size=4 << 20):
                        out.write(chunk)
                        have += len(chunk)
                        with JOBS_LOCK:
                            JOBS[model_id]["done"] = finished + have
            os.replace(part, dest)
            finished += size or have
        with JOBS_LOCK:
            JOBS[model_id] = {"state": "done", "done": finished, "total": finished}
        log.info("[weightroom] installed %s", model_id)
    except Exception as e:  # keep the .part file so the next try resumes
        log.exception("[weightroom] install of %s failed", model_id)
        with JOBS_LOCK:
            JOBS[model_id] = {**JOBS.get(model_id, {}), "state": "error", "error": str(e)[:300]}


def _pull_ollama(model_id):
    import requests
    tag = CATALOG[model_id]["ollama"]
    with JOBS_LOCK:
        JOBS[model_id] = {"state": "downloading", "done": 0, "total": 0}
    try:
        with requests.post(OLLAMA + "/api/pull", json={"model": tag, "stream": True}, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                if msg.get("total"):
                    with JOBS_LOCK:
                        JOBS[model_id].update(done=msg.get("completed", 0), total=msg["total"])
        with JOBS_LOCK:
            JOBS[model_id] = {"state": "done", "done": 1, "total": 1}
    except Exception as e:
        log.exception("[weightroom] pulling %s failed", tag)
        with JOBS_LOCK:
            JOBS[model_id] = {**JOBS.get(model_id, {}), "state": "error", "error": str(e)[:300]}


# ---------- routes ----------

routes = PromptServer.instance.routes


@routes.get("/weightroom/info")
async def info(request):
    import psutil
    import comfy.model_management as mm
    try:
        dev = mm.get_torch_device()
        gpu = {"name": mm.get_torch_device_name(dev), "vram_gb": round(mm.get_total_memory(dev) / 2**30, 1)}
    except Exception as e:
        gpu = {"name": "unknown", "vram_gb": 0, "error": str(e)[:200]}
    with JOBS_LOCK:  # read the jobs before asking Ollama, so a pull that just finished shows as installed
        jobs = {mid: dict(j) for mid, j in JOBS.items()}
    om = await _ollama_models()
    try:  # ComfyUI's INT8 "comfy kitchen attention" exists only on GPUs that support it
        import comfy.ldm.modules.attention as attn
        kitchen = bool(getattr(attn, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False))
    except Exception:
        kitchen = False
    return web.json_response({
        "bridge": VERSION,
        "gpu": gpu,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "ollama": om is not None,
        "features": {"kitchen_attention": kitchen},
        "models": {mid: {"name": e["name"], **status(mid, om, jobs.get(mid) or {})} for mid, e in CATALOG.items()},
    })


@routes.post("/weightroom/install")
async def install(request):
    body = await request.json()
    mid = body.get("model")
    if mid not in CATALOG:
        return web.json_response({"error": "Unknown model."}, status=400)
    with JOBS_LOCK:
        if (JOBS.get(mid) or {}).get("state") == "downloading":
            return web.json_response({"ok": True, "already": True})
    entry = CATALOG[mid]
    if entry.get("ollama") and await _ollama_models() is None:
        return web.json_response({"error": "Ollama is not running on this machine. Start it with the Weightroom launcher."}, status=409)
    threading.Thread(target=_pull_ollama if entry.get("ollama") else _download, args=(mid,), daemon=True).start()
    return web.json_response({"ok": True})


@routes.post("/weightroom/chat")
async def chat(request):
    body = await request.json()
    mid = body.get("model", "gpt-oss-20b")
    entry = CATALOG.get(mid)
    if not entry or not entry.get("ollama"):
        return web.json_response({"error": "That chat model is not set up for your GPU."}, status=400)
    payload = {"model": entry["ollama"], "messages": body.get("messages", [])[-40:], "stream": False}
    if entry["ollama"].startswith("gpt-oss"):
        payload["think"] = "low"
    if SHARED_GPU and not _comfy_busy():
        await asyncio.get_running_loop().run_in_executor(None, _free_comfy)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
            async with s.post(OLLAMA + "/api/chat", json=payload) as r:
                data = await r.json(content_type=None)
                if r.status != 200 or data.get("error"):
                    return web.json_response({"error": data.get("error") or f"Ollama answered {r.status}."}, status=502)
    except aiohttp.ClientConnectorError:
        return web.json_response({"error": "Ollama is not running on this machine. Start it with the Weightroom launcher."}, status=503)
    return web.json_response({"text": (data.get("message") or {}).get("content", ""), "model": entry["ollama"]})


PromptServer.instance.app.middlewares.append(key_check)
log.info("[weightroom] bridge %s ready. Key file: %s", VERSION, os.path.join(folder_paths.get_user_directory(), "weightroom_key.txt"))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
