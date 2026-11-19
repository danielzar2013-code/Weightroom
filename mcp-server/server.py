"""Weightroom MCP: lets Claude generate images, video, 3D models, speech and music with open-weight models.

Image:  Z-Image Turbo on a Hugging Face ZeroGPU Space, with Pollinations (FLUX) as a fallback.
Video:  MiniMax H3 (Turbo LoRA) on the official MiniMax Space.
3D:     Hunyuan3D 2.1 on the official Tencent Space (shape only when textures exceed the free GPU limit).
Speech: Chatterbox Multilingual V3 on the official Resemble AI Space (23 languages, Hebrew included).
Music:  ACE-Step 1.5 on the official ACE-Step Space.
Models: read live from the Weightroom app, so new models appear here too.

ZeroGPU jobs count against the daily GPU quota of the Hugging Face account whose token is in HF_TOKEN.
"""
import asyncio
import os
import re
import secrets
import time
import urllib.parse

import httpx
import uvicorn
from gradio_client import Client, handle_file
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import HTMLResponse, JSONResponse

MCP_KEY = os.environ.get("MCP_KEY", "").strip() or secrets.token_urlsafe(24)
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PORT = int(os.environ.get("PORT", "7860"))
APP_URL = "https://danielzar2013-code.github.io/Weightroom/"
APP_ORIGIN = "https://danielzar2013-code.github.io"
VERSION = 2

SPACE_IMAGE = os.environ.get("SPACE_IMAGE", "Tongyi-MAI/Z-Image-Turbo")
SPACE_VIDEO = os.environ.get("SPACE_VIDEO", "MiniMaxAI/MiniMax-H3-Turbo-Lora")
SPACE_3D = os.environ.get("SPACE_3D", "tencent/Hunyuan3D-2.1")
SPACE_TTS = os.environ.get("SPACE_TTS", "ResembleAI/Chatterbox-Multilingual-TTS-V3")
SPACE_MUSIC = os.environ.get("SPACE_MUSIC", "ACE-Step/Ace-Step-v1.5")

IMAGE_SIZES = {
    "1:1": "1024x1024 ( 1:1 )", "4:3": "1152x864 ( 4:3 )", "3:4": "864x1152 ( 3:4 )",
    "3:2": "1248x832 ( 3:2 )", "2:3": "832x1248 ( 2:3 )", "16:9": "1280x720 ( 16:9 )",
    "9:16": "720x1280 ( 9:16 )", "21:9": "1344x576 ( 21:9 )",
}
POLL_SIZES = {"1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152), "3:2": (1248, 832),
              "2:3": (832, 1248), "16:9": (1280, 720), "9:16": (720, 1280), "21:9": (1344, 576)}
VIDEO_CANVAS = {"16:9": "960x544 · 16:9 fast", "9:16": "544x960 · 9:16 fast", "1:1": "544x544 · 1:1 fast"}
TTS_LANGS = ["en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
             "pl", "pt", "ru", "sv", "sw", "tr", "zh"]
TOOLS = ["generate_image", "generate_video", "image_to_3d", "text_to_speech", "generate_music", "list_models"]

_clients: dict[str, Client] = {}


def _client(space: str) -> Client:
    if space not in _clients:
        _clients[space] = Client(space, token=HF_TOKEN, download_files=False, verbose=False)
    return _clients[space]


def _url(file_obj) -> str | None:
    """Public URL of a file returned by a Gradio Space.

    With download_files=False the client returns FileData dicts. Some Spaces return their files wrapped in a
    Gradio update ({"__type__": "update", "value": {...}}), and video components nest the file under "video".
    """
    if isinstance(file_obj, dict):
        for key in ("value", "video", "image"):
            if isinstance(file_obj.get(key), dict):
                return _url(file_obj[key])
        return file_obj.get("url") or file_obj.get("path")
    if isinstance(file_obj, str):
        return file_obj
    return None


async def _predict(space: str, api_name: str, timeout: float, **kwargs):
    def run():
        return _client(space).predict(api_name=api_name, **kwargs)
    return await asyncio.wait_for(asyncio.to_thread(run), timeout=timeout)


def _friendly(err: Exception) -> str:
    text = str(err)
    low = text.lower()
    if "quota" in low or "runs limit" in low or "exceeded your" in low:
        return ("The free Hugging Face GPU minutes for today are used up. They refill over the day. "
                "A Hugging Face PRO account gets about 8 times more.")
    if "maximum allowed" in low:
        return "This job needs more GPU time than the free Hugging Face tier allows."
    if isinstance(err, asyncio.TimeoutError):
        return "The model took too long to answer. The shared GPU queue may be busy; try again in a minute."
    if "sleep" in low or "paused" in low or "building" in low:
        return "That model Space is waking up. Try again in a minute."
    return f"The model Space returned an error: {text[:300]}"


server = MCPServer(
    name="Weightroom",
    instructions=(
        "Weightroom generates media with open-weight models. Use generate_image for pictures, generate_video for "
        "short clips with sound, image_to_3d to turn a picture into a 3D model, text_to_speech for voice in 23 "
        "languages including Hebrew, generate_music for songs, and list_models to compare models and the GPU memory "
        "they need. Write image, video and music prompts in English even when the user writes in another language, "
        "but keep text_to_speech text in the user's language. Video, 3D and music jobs take one to three minutes "
        "and use the owner's free daily GPU minutes, so ask before running many of them. Always give the user the "
        "returned links."
    ),
)


@server.tool(title="Generate an image")
async def generate_image(prompt: str, aspect_ratio: str = "1:1") -> list:
    """Generate one image from a text prompt with Z-Image Turbo, an open 6B model (falls back to FLUX on Pollinations).

    Args:
        prompt: What the image should show, in English: subject, setting, light and style.
        aspect_ratio: One of 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16, 21:9.
    """
    ratio = aspect_ratio if aspect_ratio in IMAGE_SIZES else "1:1"
    note = ""
    try:
        result = await _predict(SPACE_IMAGE, "/generate", 180, prompt=prompt, resolution=IMAGE_SIZES[ratio],
                                seed=42, steps=8, shift=3.0, random_seed=True, gallery_images=[])
        gallery = result[0] if isinstance(result, (list, tuple)) else result
        url = _url(gallery[0]) if isinstance(gallery, list) and gallery else _url(gallery)
        if not url:
            raise RuntimeError("no image in the response")
        model = "Z-Image Turbo"
    except Exception as err:  # fall back to Pollinations, which needs no GPU quota
        w, h = POLL_SIZES[ratio]
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) +
               f"?width={w}&height={h}&model=flux&nologo=true&seed={secrets.randbelow(100000)}")
        model = "FLUX on Pollinations"
        note = f" Z-Image Turbo was unavailable: {_friendly(err)}"
    async with httpx.AsyncClient(timeout=150, follow_redirects=True) as http:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN and ".hf.space" in url else {}
        resp = await http.get(url, headers=headers)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "image/png")
    fmt = "jpeg" if "jpeg" in ctype or "jpg" in ctype else "webp" if "webp" in ctype else "png"
    return [Image(data=resp.content, format=fmt), f"Made with {model}, {ratio}.{note}\nImage link: {url}"]


@server.tool(title="Generate a video")
async def generate_video(prompt: str, aspect_ratio: str = "16:9", seconds: int = 5, first_frame_url: str = "") -> str:
    """Generate a short video with sound using MiniMax H3, the top open video model. Takes one to three minutes.

    Args:
        prompt: The shot in English: subject, action, camera movement, light, and the sound you want.
        aspect_ratio: One of 16:9, 9:16, 1:1.
        seconds: Length from 2 to 14 seconds. Longer clips use more of the daily GPU minutes.
        first_frame_url: Optional public image URL to animate as the first frame.
    """
    canvas = VIDEO_CANVAS.get(aspect_ratio, VIDEO_CANVAS["16:9"])
    seconds = max(2, min(14, int(seconds)))
    first = handle_file(first_frame_url) if first_frame_url else None
    try:
        out = await _predict(SPACE_VIDEO, "/output_video", 480, in_0=prompt, in_1=first, in_2=None, in_3=canvas,
                             in_4=float(seconds), in_5=6.0, in_6=float(secrets.randbelow(100000)), in_7=False,
                             in_8="larry")
    except Exception as err:
        return _friendly(err)
    url = _url(out[0]) if isinstance(out, (list, tuple)) else _url(out)
    if not url:
        return "The video model finished but returned no file. Try again."
    return f"Video ready ({seconds}s, {aspect_ratio}, MiniMax H3 with sound): {url}"


@server.tool(title="Turn an image into a 3D model")
async def image_to_3d(image_url: str, textured: bool = True) -> str:
    """Turn one picture of an object into a 3D model (GLB) with Hunyuan3D 2.1. Takes one to three minutes.

    Args:
        image_url: Public URL of a picture showing one object, ideally on a plain background. To make a 3D model
            from a description, first call generate_image and pass its image link here.
        textured: True to try a textured mesh (needs more GPU time than the free tier allows, so it may come back
            as shape only), False for shape only.
    """
    kwargs = dict(steps=30, guidance_scale=5.0, seed=1234, octree_resolution=256, check_box_rembg=True,
                  num_chunks=8000, randomize_seed=True)
    note = ""
    out = None
    if textured:
        try:
            out = await _predict(SPACE_3D, "/generation_all", 480, image=handle_file(image_url), **kwargs)
        except Exception as err:
            if "maximum allowed" not in str(err).lower():
                return _friendly(err)
            textured = False
            note = " Textures need more GPU time than free Hugging Face accounts get, so this one is shape only."
    if out is None:
        try:
            out = await _predict(SPACE_3D, "/shape_generation", 480, image=handle_file(image_url), **kwargs)
        except Exception as err:
            return _friendly(err)
    items = list(out) if isinstance(out, (list, tuple)) else [out]
    url = _url(items[1]) if textured and len(items) > 1 and _url(items[1]) else _url(items[0])
    if not url or not str(url).startswith("http"):
        return "The 3D model finished but returned no file. Try again."
    return f"3D model ready (Hunyuan3D 2.1, {'textured' if textured else 'shape only'}, GLB): {url}{note}"


@server.tool(title="Text to speech")
async def text_to_speech(text: str, language: str = "he", expressiveness: float = 0.5) -> str:
    """Read text aloud with Chatterbox Multilingual V3. Supports 23 languages including Hebrew (he).

    Args:
        text: Up to 300 characters to speak, in the chosen language.
        language: Language code: he, en, ar, da, de, el, es, fi, fr, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw,
            tr or zh.
        expressiveness: 0.25 calm to 1.0 dramatic; 0.5 is neutral.
    """
    lang = language if language in TTS_LANGS else "en"
    try:
        out = await _predict(SPACE_TTS, "/generate_tts_audio", 240, text_input=text[:300], audio_prompt_path_input=None,
                             language_id_input=lang,
                             exaggeration_input=float(max(0.25, min(1.0, expressiveness))), temperature_input=0.8,
                             seed_num_input=0, cfgw_input=0.5)
    except Exception as err:
        return _friendly(err)
    url = _url(out[0] if isinstance(out, (list, tuple)) else out)
    return f"Audio ready (Chatterbox Multilingual V3, {lang}): {url}" if url else "No audio came back. Try again."


@server.tool(title="Generate music")
async def generate_music(style: str, lyrics: str = "", seconds: int = 60) -> str:
    """Make a song or an instrumental with ACE-Step 1.5. Takes one to three minutes.

    Args:
        style: Comma-separated style tags in English: genre, instruments, tempo, mood, vocal type.
        lyrics: Song lyrics, with [verse] and [chorus] markers. Any language, Hebrew included. Leave empty for an
            instrumental.
        seconds: Length: 30, 60, 120 or 240.
    """
    secs = min((30, 60, 120, 240), key=lambda s: abs(s - int(seconds)))
    words = lyrics.strip() or "[Instrumental]"
    lang = "he" if re.search(r"[֐-׿]", words) else "unknown"
    try:
        out = await _predict(SPACE_MUSIC, "/generation_wrapper", 480, selected_model="acestep-v15-turbo",
                             generation_mode="custom", simple_query_input="", simple_vocal_language="unknown",
                             param_4=style, param_5=words, param_9=lang, param_14=None, param_15=secs, param_16=1,
                             param_17=None, param_18="", param_47="vocals")
    except Exception as err:
        return _friendly(err)
    url = _url(out[0] if isinstance(out, (list, tuple)) else out)
    kind = "instrumental" if words == "[Instrumental]" else "song"
    return f"Music ready (ACE-Step 1.5, {secs}s {kind}): {url}" if url else "No music came back. Try again."


_models_cache: dict = {"at": 0.0, "rows": []}
_ENTRY = re.compile(r"\{id:'([^']+)',cat:'([^']+)',name:'([^']+)',maker:'([^']*)',params:'([^']*)',date:'([^']*)',"
                    r"variants:\[([^\]]*)\],lic:'([^']+)',license:'([^']*)'")
_VARIANT = re.compile(r"\{p:'([^']*)',gb:(\d+(?:\.\d+)?)\}")


async def _library() -> list[dict]:
    """The model library, read from the Weightroom app so it always matches the app."""
    if time.time() - _models_cache["at"] < 3600 and _models_cache["rows"]:
        return _models_cache["rows"]
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
        html = (await http.get(APP_URL)).text
    rows = []
    for m in _ENTRY.finditer(html):
        mid, cat, name, maker, params, date, variants, lic, license_ = m.groups()
        rows.append({"cat": cat, "name": name, "maker": maker, "params": params, "date": date, "lic": lic,
                     "license": license_.replace("\\'", "'"),
                     "variants": [(p, float(gb)) for p, gb in _VARIANT.findall(variants)]})
    if rows:
        _models_cache.update(at=time.time(), rows=rows)
    return rows


@server.tool(title="List Weightroom models")
async def list_models(category: str = "all", max_vram_gb: int = 0) -> str:
    """List the open models in the Weightroom library: maker, size, release month, license and the GPU memory
    (VRAM) each needs at each precision. Read live from the Weightroom app.

    Args:
        category: all, video, image, 3d, speech, music or chat.
        max_vram_gb: If above 0, only models whose smallest version fits in this much VRAM.
    """
    try:
        rows = await _library()
    except Exception as err:
        return f"Could not read the model library right now: {str(err)[:200]}"
    cat = category.lower().strip()
    picked = [r for r in rows if cat in ("", "all", r["cat"])]
    if max_vram_gb > 0:
        picked = [r for r in picked if r["variants"] and min(gb for _, gb in r["variants"]) <= max_vram_gb]
    if not picked:
        return "No models match."
    lic = {"open": "open license", "limited": "open weights with conditions", "nc": "non-commercial"}
    lines = [f"{r['cat']}: {r['name']} by {r['maker']} ({r['params']}, {r['date']}) — "
             + ", ".join(f"{p} ≈{gb:g} GB" for p, gb in r["variants"]) + f" — {r['license']} ({lic.get(r['lic'], r['lic'])})"
             for r in picked]
    return f"{len(picked)} models from {APP_URL}#models\n" + "\n".join(lines)


app = server.streamable_http_app(
    streamable_http_path=f"/{MCP_KEY}/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    host="0.0.0.0",
)


async def health(request):
    """Public status for the app's MCP page. Reveals nothing private: no key, no token."""
    return JSONResponse({"ok": True, "name": "Weightroom MCP", "version": VERSION, "tools": TOOLS,
                         "gpu_minutes": "account" if HF_TOKEN else "anonymous"},
                        headers={"Access-Control-Allow-Origin": APP_ORIGIN, "Cache-Control": "no-store"})


async def home(request):
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Weightroom MCP</title><body style='font:16px system-ui;background:#0b0c0a;color:#f1f2ec;"
        "display:grid;place-items:center;min-height:100vh;margin:0'><div style='max-width:36em;padding:24px'>"
        "<h1>Weightroom MCP is running</h1><p>Connect it to Claude with your private MCP address. "
        f"Instructions: <a style='color:#cdfb3a' href='{APP_URL}#mcp'>{APP_URL}#mcp</a></p></div>")

app.add_route("/health", health)
app.add_route("/", home)

if __name__ == "__main__":
    if not os.environ.get("MCP_KEY"):
        print("MCP_KEY not set; using a random path for this run:", f"/{MCP_KEY}/mcp", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
