# Chatterbox Studio

Chatterbox Studio is a polished local production interface for Resemble AI's
open-source Chatterbox models. It keeps voice generation private on your own
machine while making long-form and batch work feel approachable.

## What it adds

- Multilingual V3 as the high-quality default, with Turbo available for speed
  and native performance tags
- MP3 or lossless WAV output
- Voice cloning from an uploaded or recorded reference clip
- Long-document chunking for full `.txt` files
- A multi-file batch queue
- Precise `[pause]`, `[pause 1.5s]`, and `[pause 500ms]` timing markers
- Turbo sound and style tag shortcuts
- Permanent, collision-safe output files in `generated_audio/`
- A responsive dark interface designed for desktop and smaller screens

## Run it locally

Chatterbox works best with an NVIDIA GPU. These Windows commands create an
isolated environment and install the CUDA 12.4 build of PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
python gradio_tts_turbo_app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

MP3 export requires `ffmpeg` to be available on your system path. On Windows,
one option is:

```powershell
winget install --id Gyan.FFmpeg -e
```

## A simple workflow

1. Keep **Multilingual V3** selected for the most natural voice quality.
2. Upload 8–20 seconds of clean reference audio from a single speaker.
3. Write or paste a script in **Create**, using pause markers where needed.
4. Select MP3 for convenient sharing or WAV for a lossless master.
5. Generate and listen. Every finished file is also saved in
   `generated_audio/`.

For books, chapters, or a folder of scripts, open **Batch queue**, upload
multiple `.txt` files, and process them in order. The first queued file also
opens in the editor for a quick review.

## Pause and performance syntax

Pause markers work with both V3 and Turbo:

```text
This thought needs a moment. [pause] Now we continue.
Hold for emphasis. [pause 2.5s] Then land the final line.
Just a beat. [pause 350ms] Perfect.
```

Turbo also supports native performance tags such as `[laugh]`, `[sigh]`,
`[whispering]`, and `[dramatic]`. The interface reveals these shortcuts when
Turbo is selected. V3 does not natively support those tags, so the app safely
removes them before V3 synthesis.

## Output and privacy

The interface launches with `share=False`; it is not exposed through a Gradio
public link. Generated audio, reference clips, logs, and the local environment
are ignored by Git so they are not accidentally published.

Chatterbox itself is created and maintained by
[Resemble AI](https://github.com/resemble-ai/chatterbox). This interface builds
on their open-source project and retains its original MIT license.
