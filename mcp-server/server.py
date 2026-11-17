"""Weightroom MCP: lets Claude generate images, video, 3D models and speech with open-weight models.

Image: Z-Image Turbo on a Hugging Face ZeroGPU Space, with Pollinations (FLUX) as a fallback.
Video: MiniMax H3 (Turbo LoRA) on the official MiniMax Space.
3D:    Hunyuan3D 2.1 on the official Tencent Space.
Speech: Chatterbox Multilingual V3 on the official Resemble AI Space (23 languages, Hebrew included).

ZeroGPU jobs count against the daily GPU quota of the Hugging Face account whose token is in HF_TOKEN.
"""
import asyncio
import os
import secrets
import urllib.parse

import httpx
import uvicorn
from gradio_client import Client, handle_file
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse

MCP_KEY = os.environ.get("MCP_KEY", "").strip() or secrets.token_urlsafe(24)
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PORT = int(os.environ.get("PORT", "7860"))

SPACE_IMAGE = os.environ.get("SPACE_IMAGE", "Tongyi-MAI/Z-Image-Turbo")
SPACE_VIDEO = os.environ.get("SPACE_VIDEO", "MiniMaxAI/MiniMax-H3-Turbo-Lora")
SPACE_3D = os.environ.get("SPACE_3D", "tencent/Hunyuan3D-2.1")
SPACE_TTS = os.environ.get("SPACE_TTS", "ResembleAI/Chatterbox-Multilingual-TTS-V3")

IMAGE_SIZES = {
    "1:1": "1024x1024 ( 1:1 )", "4:3": "1152x864 ( 4:3 )", "3:4": "864x1152 ( 3:4 )",
    "3:2": "1248x832 ( 3:2 )", "2:3": "832x1248 ( 2:3 )", "16:9": "1280x720 ( 16:9 )",
    "9:16": "720x1280 ( 9:16 )", "21:9": "1344x576 ( 21:9 )",
}
POLL_SIZES = {"1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152), "3:2": (1248, 832),
              "2:3": (832, 1248), "16:9": (1280, 720), "9:16": (720, 1280), "21:9": (1344, 576)}
VIDEO_CANVAS = {"16:9": "960x544 · 16:9 fast", "9:16": "544x960 · 9:16 fast", "1:1": "544x544 · 1:1 fast",
                "4:3": "768x576 · 4:3 fast", "3:4": "576x768 · 3:4 fast", "21:9": "1152x512 · 21:9 fast"}
TTS_LANGS = ["en", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
             "pl", "pt", "ru", "sv", "sw", "tr", "zh"]

_clients: dict[str, Client] = {}


def _client(space: str) -> Client:
    if space not in _clients:
        _clients[space] = Client(space, token=HF_TOKEN, download_files=False, verbose=False)
    return _clients[space]


def _url(file_obj) -> str | None:
    """Public URL of a file returned by a Gradio Space (download_files=False returns FileData dicts)."""
    if isinstance(file_obj, dict):
        if isinstance(file_obj.get("video"), dict):
            return _url(file_obj["video"])
        if isinstance(file_obj.get("image"), dict):
            return _url(file_obj["image"])
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
    if "quota" in low or "gpu limit" in low or "exceeded" in low:
        return ("The free Hugging Face GPU quota for today is used up. It resets 24 hours after the first use. "
                "A Hugging Face PRO account gets about 8 times more.")
    if isinstance(err, asyncio.TimeoutError):
        return "The model took too long to answer. The shared GPU queue may be busy; try again in a minute."
    return f"The model Space returned an error: {text[:300]}"


server = MCPServer(
    name="Weightroom",
    instructions=(
        "Weightroom generates media with open-weight models. Use generate_image for pictures, generate_video for "
        "short clips with sound, image_to_3d to turn a picture into a 3D model, and text_to_speech for voice "
        "(Hebrew supported). Video and 3D jobs take one to three minutes and use the owner's free daily GPU quota, "
        "so ask before running many of them. Share the returned links with the user."
    ),
)


@server.tool(title="Generate an image")
async def generate_image(prompt: str, aspect_ratio: str = "1:1") -> list:
    """Generate one image from a text prompt with Z-Image Turbo, an open 6B model (falls back to FLUX on Pollinations).

    Args:
        prompt: What the image should show. English works best; describe subject, setting, light and style.
        aspect_ratio: One of 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16, 21:9.
    """
    ratio = aspect_ratio if aspect_ratio in IMAGE_SIZES else "1:1"
    note = ""
    try:
        result = await _predict(SPACE_IMAGE, "/generate", 180, prompt=prompt, resolution=IMAGE_SIZES[ratio],
                                seed=42, steps=8, shift=3.0, random_seed=True, gallery_images=[])
        gallery = result[0] if isinstance(result, (list, tuple)) else result
        url = _url(gallery[0]) if gallery else None
        if not url:
            raise RuntimeError("no image in the response")
        model = "Z-Image Turbo"
    except Exception as err:  # fall back to Pollinations, which needs no GPU quota
        w, h = POLL_SIZES[ratio]
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) +
               f"?width={w}&height={h}&model=flux&nologo=true")
        model = "FLUX on Pollinations"
        note = f" (Z-Image Turbo was unavailable: {_friendly(err)})"
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
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
        prompt: The shot: subject, action, camera movement, light, and the sound you want.
        aspect_ratio: One of 16:9, 9:16, 1:1, 4:3, 3:4, 21:9.
        seconds: Length from 2 to 14 seconds. Longer clips use more of the daily GPU quota.
        first_frame_url: Optional public image URL to animate as the first frame.
    """
    canvas = VIDEO_CANVAS.get(aspect_ratio, VIDEO_CANVAS["16:9"])
    seconds = max(2, min(14, int(seconds)))
    first = handle_file(first_frame_url) if first_frame_url else None
    try:
        out = await _predict(SPACE_VIDEO, "/output_video", 420, in_0=prompt, in_1=first, in_2=None, in_3=canvas,
                             in_4=float(seconds), in_5=6.0, in_6=42.0, in_7=False, in_8="larry")
    except Exception as err:
        return _friendly(err)
    url = _url(out[0]) if isinstance(out, (list, tuple)) else _url(out)
    report = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else ""
    return f"Video ready ({seconds}s, {aspect_ratio}, MiniMax H3 with sound): {url}\n{report}".strip()


@server.tool(title="Turn an image into a 3D model")
async def image_to_3d(image_url: str, textured: bool = True) -> str:
    """Turn one picture of an object into a 3D model (GLB) with Hunyuan3D 2.1. Takes one to three minutes.

    Args:
        image_url: Public URL of a picture showing one object, ideally on a plain background.
        textured: True for a textured mesh, False for shape only (faster).
    """
    api = "/generation_all" if textured else "/shape_generation"
    try:
        out = await _predict(SPACE_3D, api, 420, image=handle_file(image_url), steps=30, guidance_scale=5.0,
                             seed=1234, octree_resolution=256, check_box_rembg=True, num_chunks=8000,
                             randomize_seed=True)
    except Exception as err:
        return _friendly(err)
    files = [_url(x) for x in (out if isinstance(out, (list, tuple)) else [out]) if _url(x) and isinstance(x, dict)]
    links = "\n".join(files[:2])
    return f"3D model ready (Hunyuan3D 2.1, {'textured' if textured else 'shape only'}):\n{links}"


@server.tool(title="Text to speech")
async def text_to_speech(text: str, language: str = "he", expressiveness: float = 0.5) -> str:
    """Read text aloud with Chatterbox Multilingual V3. Supports 23 languages including Hebrew (he).

    Args:
        text: Up to 300 characters to speak.
        language: Language code, for example he, en, ar, fr, de, es, ru.
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
    return f"Audio ready (Chatterbox Multilingual V3, {lang}): {_url(out)}"


@server.tool(title="List Weightroom models")
def list_models(category: str = "all") -> str:
    """List the open models in the Weightroom library with the VRAM each needs, for planning a local GPU.

    Args:
        category: all, video, image, 3d, speech, music or chat.
    """
    rows = [
        ("video", "MiniMax H3", "BF16 ≈80 GB, FP8 ≈44 GB"), ("video", "LTX-2.5", "BF16 ≈52 GB, FP8 ≈30 GB"),
        ("video", "Wan 2.2 A14B", "BF16 ≈64 GB, FP8 ≈34 GB"), ("video", "HunyuanVideo 1.5", "BF16 ≈28 GB, FP8 ≈16 GB"),
        ("image", "Qwen-Image-2.1", "BF16 ≈34 GB, FP8 ≈18 GB"), ("image", "FLUX.2 [dev]", "BF16 ≈90 GB, FP8 ≈48 GB"),
        ("image", "Z-Image Turbo", "BF16 ≈16 GB, FP8 ≈10 GB"), ("image", "Qwen-Image", "BF16 ≈58 GB, FP8 ≈30 GB"),
        ("3d", "Hunyuan3D 2.1", "≈29 GB with texture, ≈10 GB shape only"), ("3d", "TRELLIS.2", "≈24 GB"),
        ("speech", "Chatterbox Multilingual V3", "≈5 GB, speaks Hebrew"), ("speech", "Step-Audio-EditX", "≈12 GB"),
        ("music", "ACE-Step 1.5", "≈8 GB"), ("chat", "Qwen3.8-27B", "BF16 ≈58 GB, FP8 ≈32 GB, Q4 ≈19 GB"),
        ("chat", "gpt-oss-120b", "MXFP4 ≈66 GB"), ("chat", "gpt-oss-20b", "MXFP4 ≈16 GB"),
    ]
    picked = [r for r in rows if category in ("all", "", r[0])]
    return "\n".join(f"{c}: {n} — {v}" for c, n, v in picked) or "No models in that category."


app = server.streamable_http_app(
    streamable_http_path=f"/{MCP_KEY}/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    host="0.0.0.0",
)


async def home(request):
    return PlainTextResponse("Weightroom MCP server is running. Add the private MCP address you were given to Claude.")

app.add_route("/", home)

if __name__ == "__main__":
    if not os.environ.get("MCP_KEY"):
        print("MCP_KEY not set; using a random path for this run:", f"/{MCP_KEY}/mcp", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
