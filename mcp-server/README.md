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

An MCP server that gives Claude five tools: `generate_image` (Z-Image Turbo, falls back to FLUX on Pollinations),
`generate_video` (MiniMax H3 with sound), `image_to_3d` (Hunyuan3D 2.1), `text_to_speech` (Chatterbox
Multilingual V3, Hebrew included) and `list_models`.

Set two Space secrets: `HF_TOKEN` (your Hugging Face token, so GPU jobs use your free daily quota) and `MCP_KEY`
(a long random string). The MCP address is `https://<space-host>/<MCP_KEY>/mcp`. Keep it private.
