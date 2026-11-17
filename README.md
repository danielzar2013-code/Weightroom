# Weightroom Studio

An installable web app for open-weight AI models: video, image, 3D, voice, music and chat, each sized against your GPU's VRAM (up to 100 GB), with public benchmark rankings.

**Open the app:** https://danielzar2013-code.github.io/weightroom/

## What runs for real

| Category | Model | Where it runs |
|---|---|---|
| Images | FLUX | Pollinations, no account |
| Images | Z-Image Turbo | Hugging Face ZeroGPU |
| Video with sound | MiniMax H3 | Hugging Face ZeroGPU |
| 3D | Hunyuan3D 2.1 | Hugging Face ZeroGPU |
| Speech, Hebrew included | Chatterbox Multilingual V3 | Hugging Face ZeroGPU |
| Chat | gpt-oss-20b | Pollinations, no account |

Hugging Face jobs use free GPU minutes. Add your own Hugging Face token in **Settings** for more; it stays on your device.

## Install

- **Windows:** open the link in Edge or Chrome, then click **Install** in the address bar. It appears in the Start menu.
- **Android:** open the link in Chrome, then **Install app** in the menu.
- **iPhone:** Share, then **Add to Home Screen**.

## MCP server

`mcp-server/` holds an MCP server that lets Claude use the same tools. See `mcp-server/README.md`.
