# Weightroom GPU

Runs the Weightroom studios on your own NVIDIA card (16 GB or more) or on a free Kaggle GPU, instead of Hugging Face's daily minutes.

It installs [ComfyUI](https://github.com/Comfy-Org/ComfyUI) `v0.45.5` with two add-ons, [Ollama](https://ollama.com) for chat, and a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) so your phone can reach it. Then it prints a link that opens the app already connected.

- `weightroom_bridge/` is the Weightroom add-on for ComfyUI:
  - It locks every request behind this machine's key, so the tunnel address alone is useless.
  - It reports the GPU and installed models.
  - It downloads models from a fixed list.
  - It forwards chat to Ollama.
  - On a single GPU, it frees memory between chat and generation.
- The other add-on is [ComfyUI_Fill-ChatterBox](https://github.com/filliptm/ComfyUI_Fill-ChatterBox), for speech in 23 languages, Hebrew included.

## On a Windows PC

1. Download [`Weightroom-GPU.bat`](https://danielzar2013-code.github.io/Weightroom/gpu/Weightroom-GPU.bat) and double-click it. If Windows warns you, choose **More info**, then **Run anyway**.
2. The first run installs everything into `%LOCALAPPDATA%\WeightroomGPU`. It takes 15 to 30 minutes and needs no admin rights.
3. Then the app opens connected. A QR code in the window connects your phone.
4. Keep the window open while you work. Close it to stop. Run the same file to start again; the link changes each time.

## On Kaggle

1. Create a notebook. In its settings, choose **GPU T4 x2** and turn **Internet** on (this needs a verified phone number).
2. Paste this into a cell and run it:

   ```
   !curl -fsSL https://danielzar2013-code.github.io/Weightroom/gpu/linux.sh -o /tmp/wr.sh && bash /tmp/wr.sh
   ```

3. Open the link it prints, or scan the QR code. ComfyUI uses the first T4 and chat uses the second.
4. To keep your videos between sessions, set **Persistence** to **Files only** in the notebook's session options, and stop the session with the stop button. Videos and pictures are saved in `/kaggle/working/weightroom-output`. Models are not kept: they download again each session.

`linux.sh` also works on any Linux machine with an NVIDIA GPU.

## Models

Install each model from the app: **Settings**, then **Your GPU**.

| Studio | Runs on your GPU | Download |
|---|---|---|
| Image | Z-Image Turbo | 21 GB |
| Video | MiniMax H3 Singularity V2 (with sound, 480p or 768p; 8-step turbo LoRA, Sol-Attn sparse attention, H3 latent upscaler for 768p; needs 32 GB+ RAM) | 34 GB |
| Video, lighter | Wan 2.2 5B (5 s, up to 720p, no sound), used when Singularity isn't installed | 18 GB |
| 3D | Hunyuan3D 2.1 (shape) | 7.4 GB |
| Speech | Chatterbox Multilingual | 3.2 GB |
| Music | ACE-Step 1.5 | 10 GB |
| Chat | gpt-oss-20b (Ollama) | 13 GB |

Other library models run as the model above for their studio, and the result card says so. The original MiniMax H3 and LTX-2.5 need more than 16 GB; on your GPU, MiniMax H3 runs as Singularity V2 (a pruned W4A8 build that fits).
