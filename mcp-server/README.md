---
title: Weightroom MCP
emoji: 🏋️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: MCP server that lets Claude generate with open models
---

# Weightroom MCP

An MCP server that gives Claude six tools:
- `generate_image`: Z-Image Turbo, falling back to FLUX on Pollinations.
- `generate_video`: MiniMax H3 with sound.
- `image_to_3d`: Hunyuan3D 2.1. It returns shape only when textures exceed the free GPU limit.
- `text_to_speech`: Chatterbox Multilingual V3, Hebrew included.
- `generate_music`: ACE-Step 1.5.
- `list_models`: the live Weightroom library, read from the app.

Set two Space secrets:
- `MCP_KEY`: a long random string. The MCP address is `https://<space-host>/<MCP_KEY>/mcp`. Keep it private.
- `HF_TOKEN`: optional. With it, GPU jobs use your account's free daily minutes instead of the shared anonymous ones.

`/health` is public and reports only that the server is up and which tools it has.

Connection steps for Claude, Claude Code and other apps are on the app's MCP page: https://danielzar2013-code.github.io/Weightroom/#mcp
