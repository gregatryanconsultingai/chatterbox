import random
import re
import shutil
import subprocess
import textwrap
from html import escape
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_CHARS = 300
CHUNK_PAUSE_SECONDS = 0.25
DEFAULT_EXPLICIT_PAUSE_SECONDS = 1.0
MAX_EXPLICIT_PAUSE_SECONDS = 30.0
MODEL_V3 = "Multilingual V3 — highest quality (recommended)"
MODEL_TURBO = "Turbo — fastest + sound/style tags"
FORMAT_MP3 = "MP3 — 160 kbps (maximum quality at 24 kHz)"
FORMAT_WAV = "WAV — lossless"
APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
OUTPUT_DIR = APP_DIR / "generated_audio"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAVICON_PATH = ASSET_DIR / "cbx-49.png"
FFMPEG_EXE = shutil.which("ffmpeg")

TURBO_EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]
TURBO_STYLE_TAGS = [
    "[angry]", "[fear]", "[surprised]", "[whispering]", "[crying]",
    "[happy]", "[sarcastic]", "[dramatic]", "[narration]", "[advertisement]",
]
TURBO_TAGS = TURBO_EVENT_TAGS + TURBO_STYLE_TAGS
PAUSE_BUTTONS = ["[pause]", "[pause 1s]", "[pause 2s]"]
PAUSE_TAG_PATTERN = re.compile(
    r"\[pause(?:\s+([0-9]*\.?[0-9]+)\s*(ms|s)?)?\]",
    re.IGNORECASE,
)
MODEL_CACHE = {}


def ensure_favicon():
    if FAVICON_PATH.exists():
        return
    icon = Image.new("RGBA", (128, 128), "#050505")
    draw = ImageDraw.Draw(icon)
    draw.polygon(
        [(4, 4), (101, 4), (124, 27), (124, 124), (4, 124)],
        outline="#ff6518",
        width=4,
    )
    draw.ellipse((30, 18, 98, 86), fill="#ff6518")
    for stripe_y in range(31, 82, 11):
        draw.rectangle((25, stripe_y, 103, stripe_y + 4), fill="#541203")
    draw.line(
        [(19, 102), (48, 102), (56, 94), (66, 110), (78, 87), (88, 102), (110, 102)],
        fill="#00dff5",
        width=3,
    )
    icon.save(FAVICON_PATH, "PNG", optimize=True)


ensure_favicon()

CUSTOM_CSS = """
:root,
.dark {
    --body-background-fill: #070914;
    --body-text-color: #eef2ff;
    --background-fill-primary: rgba(13, 17, 35, 0.92);
    --background-fill-secondary: rgba(19, 24, 46, 0.82);
    --block-background-fill: rgba(14, 18, 37, 0.86);
    --block-border-color: rgba(148, 163, 184, 0.16);
    --block-label-background-fill: transparent;
    --block-title-text-color: #f8fafc;
    --block-label-text-color: #a5b4fc;
    --input-background-fill: rgba(4, 7, 20, 0.72);
    --input-border-color: rgba(148, 163, 184, 0.20);
    --input-placeholder-color: #64748b;
    --border-color-primary: rgba(148, 163, 184, 0.18);
    --color-accent: #8b5cf6;
    --color-accent-soft: rgba(139, 92, 246, 0.15);
    --shadow-drop: 0 28px 80px rgba(0, 0, 0, 0.28);
}

html,
body {
    background: #070914 !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 28px 56px !important;
    color: #e5e7eb !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif !important;
    background:
        radial-gradient(circle at 12% -8%, rgba(124, 58, 237, 0.25), transparent 34%),
        radial-gradient(circle at 88% 4%, rgba(6, 182, 212, 0.16), transparent 29%),
        radial-gradient(circle at 52% 108%, rgba(236, 72, 153, 0.10), transparent 30%),
        #070914 !important;
}

footer,
.footer,
.built-with {
    display: none !important;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 28px;
    padding: 36px 38px;
    margin-bottom: 22px;
    border: 1px solid rgba(167, 139, 250, 0.20);
    border-radius: 30px;
    background:
        linear-gradient(115deg, rgba(124, 58, 237, 0.20), rgba(15, 23, 42, 0.80) 52%, rgba(8, 145, 178, 0.14)),
        rgba(10, 13, 28, 0.90);
    box-shadow: 0 32px 90px rgba(0, 0, 0, 0.36);
}

.hero-shell::before {
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    right: 14%;
    top: -190px;
    border-radius: 999px;
    background: #8b5cf6;
    filter: blur(90px);
    opacity: 0.42;
    pointer-events: none;
}

.hero-copy,
.hero-meta {
    position: relative;
    z-index: 1;
}

.brand-mark {
    display: inline-grid;
    place-items: center;
    width: 42px;
    height: 42px;
    margin-bottom: 20px;
    border: 1px solid rgba(196, 181, 253, 0.40);
    border-radius: 14px;
    color: #fff;
    font-size: 21px;
    background: linear-gradient(145deg, #8b5cf6, #4f46e5);
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.40);
}

.eyebrow,
.section-eyebrow {
    margin: 0 0 8px;
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.hero-shell h1 {
    margin: 0;
    color: #ffffff;
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 780;
    letter-spacing: -0.052em;
    line-height: 0.96;
}

.hero-shell h1 span {
    color: transparent;
    background: linear-gradient(100deg, #c4b5fd 5%, #67e8f9 94%);
    background-clip: text;
    -webkit-background-clip: text;
}

.hero-subtitle {
    max-width: 660px;
    margin: 16px 0 0;
    color: #aeb8ce;
    font-size: 16px;
    line-height: 1.65;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 9px;
    min-width: 280px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 9px 12px;
    border: 1px solid rgba(148, 163, 184, 0.17);
    border-radius: 999px;
    color: #dbeafe;
    font-size: 12px;
    font-weight: 650;
    white-space: nowrap;
    background: rgba(3, 7, 18, 0.48);
    backdrop-filter: blur(18px);
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #34d399;
    box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.12), 0 0 18px #34d399;
}

.studio-card {
    overflow: hidden !important;
    padding: 22px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 24px !important;
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.025), transparent 55%),
        rgba(13, 17, 35, 0.82) !important;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.20) !important;
    backdrop-filter: blur(20px);
}

.studio-card:hover {
    border-color: rgba(167, 139, 250, 0.23) !important;
}

.section-title {
    margin: 0 0 5px !important;
    color: #f8fafc !important;
    font-size: 21px !important;
    font-weight: 720 !important;
    letter-spacing: -0.025em !important;
}

.section-copy {
    margin: 0 0 18px !important;
    color: #8793aa !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
}

.settings-row {
    margin-bottom: 20px !important;
}

.model-status {
    min-height: 0 !important;
    padding: 12px 14px !important;
    border: 1px solid rgba(129, 140, 248, 0.18) !important;
    border-radius: 14px !important;
    color: #c7d2fe !important;
    background: rgba(79, 70, 229, 0.10) !important;
}

.model-status p {
    margin: 0 !important;
    font-size: 12.5px !important;
    line-height: 1.55 !important;
}

.tabs-shell {
    border: 0 !important;
    background: transparent !important;
}

.tabs-shell > .tab-nav {
    width: fit-content;
    gap: 6px;
    padding: 5px !important;
    margin: 0 0 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 15px !important;
    background: rgba(8, 11, 25, 0.72) !important;
}

.tabs-shell > .tab-nav button {
    padding: 9px 16px !important;
    border: 0 !important;
    border-radius: 10px !important;
    color: #8190aa !important;
    font-weight: 700 !important;
}

.tabs-shell > .tab-nav button.selected {
    color: #fff !important;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.88), rgba(79, 70, 229, 0.88)) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25) !important;
}

#main_textbox textarea {
    min-height: 330px !important;
    padding: 18px !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    caret-color: #a78bfa;
}

#main_textbox textarea:focus {
    box-shadow: 0 0 0 1px #8b5cf6, 0 0 0 5px rgba(139, 92, 246, 0.11) !important;
}

.tag-heading {
    margin: 12px 0 0 !important;
    color: #93a0b7 !important;
    font-size: 12px !important;
}

.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin: 3px 0 10px !important;
    border: none !important;
    background: transparent !important;
}

.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 31px !important;
    padding: 0 11px !important;
    margin: 0 !important;
    border: 1px solid rgba(139, 92, 246, 0.23) !important;
    border-radius: 999px !important;
    color: #c4b5fd !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    background: rgba(124, 58, 237, 0.09) !important;
    box-shadow: none !important;
    transition: all 160ms ease !important;
}

.tag-btn:hover {
    border-color: rgba(167, 139, 250, 0.55) !important;
    color: #fff !important;
    background: rgba(124, 58, 237, 0.22) !important;
    transform: translateY(-2px);
}

.action-row {
    gap: 10px !important;
    margin-top: 6px !important;
}

#generate-btn,
#queue-btn {
    min-height: 48px !important;
    border: 0 !important;
    border-radius: 14px !important;
    font-weight: 750 !important;
    letter-spacing: -0.01em !important;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease !important;
}

#generate-btn {
    color: #fff !important;
    background: linear-gradient(105deg, #7c3aed, #4f46e5 55%, #0891b2) !important;
    box-shadow: 0 14px 34px rgba(99, 102, 241, 0.28) !important;
}

#queue-btn {
    color: #fff !important;
    background: linear-gradient(105deg, #7c3aed, #4f46e5) !important;
    box-shadow: 0 14px 34px rgba(99, 102, 241, 0.24) !important;
}

#generate-btn:hover,
#queue-btn:hover {
    filter: brightness(1.12);
    transform: translateY(-2px);
}

.output-card audio {
    border-radius: 15px !important;
}

.output-note {
    padding: 13px 15px !important;
    border: 1px solid rgba(34, 211, 238, 0.14) !important;
    border-radius: 14px !important;
    background: rgba(8, 145, 178, 0.08) !important;
}

.output-note p,
.queue-copy p {
    color: #9aa9c1 !important;
    font-size: 12.5px !important;
    line-height: 1.6 !important;
}

.output-note code {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

.advanced-card {
    margin-top: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.13) !important;
    border-radius: 17px !important;
    background: rgba(7, 10, 24, 0.42) !important;
}

.advanced-card > .label-wrap {
    padding: 14px 16px !important;
}

.voice-tip {
    padding: 12px 14px;
    margin-top: 12px;
    border-left: 2px solid #22d3ee;
    border-radius: 0 10px 10px 0;
    color: #8fa0b9;
    font-size: 12px;
    line-height: 1.55;
    background: rgba(6, 182, 212, 0.06);
}

.queue-icon {
    display: grid;
    place-items: center;
    width: 46px;
    height: 46px;
    margin-bottom: 12px;
    border: 1px solid rgba(103, 232, 249, 0.22);
    border-radius: 15px;
    color: #67e8f9;
    font-size: 20px;
    background: rgba(8, 145, 178, 0.10);
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 16px 14px 38px !important;
    }

    .hero-shell {
        align-items: flex-start;
        flex-direction: column;
        padding: 28px 24px;
        border-radius: 23px;
    }

    .hero-meta {
        justify-content: flex-start;
        min-width: 0;
    }

    .studio-card {
        padding: 17px !important;
        border-radius: 20px !important;
    }

    #main_textbox textarea {
        min-height: 260px !important;
    }
}
"""

NEON_NOIR_CSS = """
:root,
.dark {
    --body-background-fill: #050505;
    --body-text-color: #e8e2d9;
    --background-fill-primary: #090909;
    --background-fill-secondary: #0d0c0b;
    --block-background-fill: #0a0a09;
    --block-border-color: rgba(255, 100, 28, 0.22);
    --block-label-background-fill: #070707;
    --block-title-text-color: #f7efe4;
    --block-label-text-color: #ff7a1a;
    --input-background-fill: #050505;
    --input-border-color: rgba(255, 116, 35, 0.25);
    --input-placeholder-color: #6f6860;
    --border-color-primary: rgba(255, 108, 28, 0.18);
    --color-accent: #ff5a18;
    --color-accent-soft: rgba(255, 90, 24, 0.12);
}

html,
body {
    background: #030303 !important;
}

.gradio-container {
    position: relative !important;
    max-width: 1540px !important;
    padding: 26px 34px 70px !important;
    color: #ded8cf !important;
    font-family: Bahnschrift, "DIN Alternate", "Arial Narrow", Arial, sans-serif !important;
    background:
        linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px),
        radial-gradient(circle at 82% 2%, rgba(255, 78, 14, 0.20), transparent 27%),
        radial-gradient(circle at 4% 48%, rgba(0, 213, 255, 0.07), transparent 25%),
        linear-gradient(135deg, #030303, #080706 52%, #050505) !important;
    background-size: 44px 44px, 44px 44px, auto, auto, auto !important;
}

.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 1000;
    pointer-events: none;
    opacity: 0.10;
    background: repeating-linear-gradient(
        to bottom,
        transparent 0,
        transparent 3px,
        rgba(255, 255, 255, 0.055) 4px
    );
    mix-blend-mode: overlay;
}

.hero-shell {
    position: relative;
    min-height: 420px;
    align-items: center;
    padding: 46px 54px 72px;
    margin-bottom: 18px;
    overflow: hidden;
    border: 1px solid rgba(255, 104, 25, 0.36);
    border-radius: 0;
    clip-path: polygon(0 0, calc(100% - 48px) 0, 100% 48px, 100% 100%, 34px 100%, 0 calc(100% - 34px));
    background:
        linear-gradient(90deg, rgba(2, 2, 2, 0.98) 0%, rgba(7, 6, 5, 0.94) 48%, rgba(23, 8, 2, 0.70) 100%),
        #050505;
    box-shadow: 0 38px 110px rgba(0, 0, 0, 0.58), inset 0 0 70px rgba(255, 63, 0, 0.035);
    animation: noir-arrival 620ms cubic-bezier(.16, .84, .3, 1) both;
}

.hero-shell::before {
    content: "";
    position: absolute;
    width: 1px;
    height: 100%;
    top: 0;
    left: 48%;
    background: linear-gradient(transparent, rgba(255, 108, 28, 0.36), transparent);
    filter: none;
    opacity: 1;
}

.hero-shell::after {
    content: "CHTRBX  /  LOCAL SYNTHESIS ARRAY  /  2049";
    position: absolute;
    right: 28px;
    bottom: 22px;
    color: rgba(255, 165, 98, 0.55);
    font-family: Consolas, monospace;
    font-size: 10px;
    letter-spacing: 0.20em;
}

.hero-copy {
    width: 60%;
}

.brand-mark {
    display: inline-flex;
    width: auto;
    height: 32px;
    padding: 0 11px;
    margin-bottom: 26px;
    border: 1px solid #ff6a1a;
    border-radius: 0;
    color: #ff8a36;
    font-family: Consolas, monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    background: rgba(255, 79, 12, 0.08);
    box-shadow: 0 0 26px rgba(255, 78, 14, 0.14);
}

.eyebrow,
.section-eyebrow {
    color: #ff6a1a;
    font-family: Consolas, monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.22em;
}

.hero-shell h1 {
    max-width: 760px;
    color: #f4ede4;
    font-size: clamp(58px, 7.6vw, 112px);
    font-weight: 840;
    line-height: 0.79;
    letter-spacing: -0.065em;
    text-transform: uppercase;
    text-shadow: 0 4px 44px rgba(0, 0, 0, 0.75);
}

.hero-shell h1 span {
    color: transparent;
    background: linear-gradient(92deg, #ff4d0b, #ff9a42 72%, #ffd0a3);
    background-clip: text;
    -webkit-background-clip: text;
    filter: drop-shadow(0 0 22px rgba(255, 74, 10, 0.18));
}

.hero-subtitle {
    max-width: 560px;
    margin-top: 24px;
    color: #9e9489;
    font-family: Consolas, monospace;
    font-size: 13px;
    line-height: 1.75;
    letter-spacing: 0.035em;
}

.hero-visual {
    position: absolute;
    right: 4.5%;
    top: 50%;
    width: min(38vw, 490px);
    aspect-ratio: 1.25;
    transform: translateY(-51%);
}

.noir-sun {
    position: absolute;
    width: 260px;
    height: 260px;
    right: 12%;
    top: 9%;
    border-radius: 50%;
    background:
        repeating-linear-gradient(to bottom, #ff7a1a 0 11px, #d73807 11px 15px, transparent 15px 20px),
        linear-gradient(#ffb05b, #fa3a07);
    box-shadow: 0 0 80px rgba(255, 68, 5, 0.42), 0 0 190px rgba(255, 68, 5, 0.16);
    animation: sun-breathe 5s ease-in-out infinite;
}

.horizon-line {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 27%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff5714 22%, #ffc087 70%, transparent);
    box-shadow: 0 0 18px #ff4a0b;
}

.horizon-line::after {
    content: "";
    position: absolute;
    inset: 1px 2% auto;
    height: 115px;
    opacity: 0.22;
    transform: perspective(120px) rotateX(62deg);
    transform-origin: top;
    background:
        repeating-linear-gradient(90deg, transparent 0 35px, rgba(255, 117, 43, 0.42) 36px 37px),
        repeating-linear-gradient(to bottom, rgba(255, 117, 43, 0.30) 0 1px, transparent 1px 18px);
}

.visual-code {
    position: absolute;
    top: 9px;
    right: 0;
    color: rgba(0, 229, 255, 0.75);
    font-family: Consolas, monospace;
    font-size: 9px;
    line-height: 1.7;
    letter-spacing: 0.12em;
    text-align: right;
}

.hero-meta {
    position: absolute;
    left: 54px;
    right: 54px;
    bottom: 24px;
    justify-content: flex-start;
    gap: 0;
    min-width: 0;
    border-top: 1px solid rgba(255, 110, 30, 0.18);
}

.status-pill {
    padding: 11px 16px 0 0;
    margin-right: 16px;
    border: 0;
    border-radius: 0;
    color: #9ea6a5;
    font-family: Consolas, monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.14em;
    background: transparent;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #00e4ff;
    box-shadow: 0 0 0 3px rgba(0, 228, 255, 0.10), 0 0 14px #00e4ff;
    animation: signal-blink 1.9s ease-in-out infinite;
}

.settings-row {
    gap: 14px !important;
    margin-bottom: 16px !important;
}

.studio-card {
    position: relative;
    padding: 23px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 107, 28, 0.19) !important;
    border-left: 2px solid rgba(255, 95, 20, 0.62) !important;
    border-radius: 0 !important;
    clip-path: polygon(0 0, calc(100% - 19px) 0, 100% 19px, 100% 100%, 0 100%);
    background:
        linear-gradient(125deg, rgba(255, 70, 6, 0.045), transparent 34%),
        rgba(8, 8, 7, 0.94) !important;
    box-shadow: 14px 20px 55px rgba(0, 0, 0, 0.34) !important;
    backdrop-filter: blur(14px);
    transition: border-color 180ms ease, transform 180ms ease !important;
}

.studio-card::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 46px;
    height: 1px;
    background: #00daf5;
    box-shadow: 0 0 12px rgba(0, 218, 245, 0.65);
}

.studio-card:hover {
    border-color: rgba(255, 111, 35, 0.40) !important;
    transform: translateY(-2px);
}

.studio-card > .block,
.studio-card > div {
    background-color: transparent !important;
}

.section-title {
    color: #efe9e1 !important;
    font-size: 24px !important;
    font-weight: 790 !important;
    letter-spacing: -0.035em !important;
    text-transform: uppercase;
}

.section-copy {
    color: #746e67 !important;
    font-family: Consolas, monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.025em;
}

.model-status {
    padding: 12px 14px !important;
    border: 0 !important;
    border-left: 2px solid #00d8f4 !important;
    border-radius: 0 !important;
    color: #9bc7cc !important;
    background: rgba(0, 205, 232, 0.055) !important;
}

.model-status p {
    font-family: Consolas, monospace !important;
    font-size: 10px !important;
}

.tabs-shell > .tab-nav {
    width: 100%;
    gap: 0;
    padding: 0 !important;
    margin-bottom: 14px !important;
    border: 0 !important;
    border-bottom: 1px solid rgba(255, 103, 25, 0.22) !important;
    border-radius: 0 !important;
    background: transparent !important;
}

.tabs-shell > .tab-nav button {
    width: auto;
    min-width: 190px;
    padding: 14px 18px !important;
    border: 0 !important;
    border-left: 1px solid rgba(255, 108, 28, 0.17) !important;
    border-radius: 0 !important;
    color: #726c65 !important;
    font-family: Consolas, monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    background: rgba(10, 10, 9, 0.72) !important;
}

.tabs-shell > .tab-nav button.selected {
    color: #130804 !important;
    background: linear-gradient(90deg, #ff4a0a, #ff8b32) !important;
    box-shadow: 0 -8px 26px rgba(255, 68, 6, 0.11) !important;
}

.block label,
.block .label-wrap,
label span {
    font-family: Consolas, monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

input,
textarea,
select,
.wrap,
.secondary-wrap {
    border-radius: 0 !important;
}

#main_textbox textarea {
    min-height: 350px !important;
    padding: 22px !important;
    border-left: 2px solid rgba(255, 95, 15, 0.55) !important;
    border-radius: 0 !important;
    color: #eee6dc !important;
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 14px !important;
    line-height: 1.85 !important;
    background:
        linear-gradient(90deg, rgba(255, 75, 8, 0.035), transparent 22%),
        repeating-linear-gradient(to bottom, transparent 0 31px, rgba(255, 255, 255, 0.025) 32px),
        #030303 !important;
}

#main_textbox textarea:focus {
    box-shadow: inset 3px 0 0 #ff5b12, 0 0 34px rgba(255, 74, 8, 0.055) !important;
}

.tag-heading {
    color: #8c8177 !important;
    font-family: Consolas, monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.tag-btn {
    height: 30px !important;
    padding: 0 12px !important;
    border: 1px solid rgba(255, 103, 25, 0.25) !important;
    border-radius: 0 !important;
    color: #d77842 !important;
    font-family: Consolas, monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.055em;
    background: rgba(255, 74, 8, 0.045) !important;
}

.tag-btn:hover {
    border-color: #ff6b1a !important;
    color: #160803 !important;
    background: #ff6b1a !important;
    box-shadow: 0 0 22px rgba(255, 80, 10, 0.22) !important;
}

#generate-btn,
#queue-btn {
    position: relative;
    min-height: 56px !important;
    border: 1px solid #ff8a39 !important;
    border-radius: 0 !important;
    color: #160603 !important;
    font-family: Consolas, monospace !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase;
    background: linear-gradient(95deg, #ff4208, #ff8b2e 64%, #ffbd72) !important;
    box-shadow: 0 0 0 1px rgba(255, 80, 8, 0.18), 0 16px 48px rgba(255, 55, 3, 0.18) !important;
}

#generate-btn::after,
#queue-btn::after {
    content: "";
    position: absolute;
    width: 10px;
    height: 10px;
    top: 7px;
    right: 7px;
    border-top: 1px solid #210802;
    border-right: 1px solid #210802;
}

#generate-btn:hover,
#queue-btn:hover {
    filter: saturate(1.2) brightness(1.12);
    box-shadow: 0 0 34px rgba(255, 74, 5, 0.34) !important;
    transform: translateY(-2px);
}

.output-note {
    border: 0 !important;
    border-left: 2px solid #00d8ef !important;
    border-radius: 0 !important;
    background: rgba(0, 208, 235, 0.045) !important;
}

.output-note p,
.queue-copy p {
    color: #7e8c8b !important;
    font-family: Consolas, monospace !important;
    font-size: 10px !important;
}

.advanced-card {
    border-radius: 0 !important;
    background: #070707 !important;
}

.voice-tip {
    border-left-color: #ff5a12;
    border-radius: 0;
    color: #79736c;
    font-family: Consolas, monospace;
    font-size: 10px;
    background: rgba(255, 78, 10, 0.04);
}

.queue-icon {
    border: 1px solid rgba(255, 102, 25, 0.42);
    border-radius: 0;
    color: #ff6a1a;
    background: rgba(255, 80, 10, 0.06);
    box-shadow: 0 0 28px rgba(255, 67, 5, 0.08);
}

@keyframes noir-arrival {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes sun-breathe {
    0%, 100% { filter: saturate(0.92); transform: scale(1); }
    50% { filter: saturate(1.18); transform: scale(1.025); }
}

@keyframes signal-blink {
    0%, 100% { opacity: 0.45; }
    50% { opacity: 1; }
}

@media (max-width: 900px) {
    html,
    body {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .gradio-container {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        padding: 12px 12px 46px !important;
        overflow-x: hidden !important;
    }

    .hero-shell {
        width: 100%;
        min-width: 0;
        max-width: 100%;
        min-height: 470px;
        padding: 32px 25px 88px;
        contain: layout paint;
        clip-path: polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 20px 100%, 0 calc(100% - 20px));
    }

    .hero-copy {
        width: 100%;
    }

    .hero-shell h1 {
        font-size: clamp(46px, 17vw, 68px);
        letter-spacing: -0.075em;
    }

    .hero-visual {
        right: -34px;
        top: 34%;
        width: 230px;
        opacity: 0.38;
    }

    .noir-sun {
        width: 190px;
        height: 190px;
    }

    .hero-meta {
        left: 25px;
        right: 25px;
        flex-wrap: wrap;
    }

    .hero-shell::after {
        display: none;
    }

    .studio-card {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        padding: 17px !important;
    }

    .settings-row,
    .tabs-shell .row:not(.tag-container):not(.action-row) {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        flex-direction: column !important;
    }

    .settings-row > .column,
    .tabs-shell .column,
    .studio-card .row:not(.tag-container):not(.action-row) > .block {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        flex: 1 1 auto !important;
    }

    .studio-card .row:not(.tag-container):not(.action-row) {
        flex-direction: column !important;
    }

    .tabs-shell > .tab-nav button {
        flex: 1;
        min-width: 0;
    }
}

@media (prefers-reduced-motion: reduce) {
    .hero-shell,
    .noir-sun,
    .status-dot {
        animation: none !important;
    }
}
"""

# Gradio scopes launch CSS to the content canvas. This small global rule keeps
# the outer application shell fluid at phone-sized breakpoints.
GLOBAL_HEAD_STYLE = """
<style>
@media (max-width: 900px) {
    html,
    body,
    gradio-app {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    gradio-app > .gradio-container {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        padding: 12px !important;
        box-sizing: border-box !important;
        overflow-x: hidden !important;
    }

    gradio-app > .gradio-container > .main {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    gradio-app > .gradio-container > .main > .wrap,
    gradio-app main.contain {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
}
</style>
"""

DECK_CSS = """
#boot-sequence {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: grid;
    place-items: center;
    color: #ff7a27;
    background:
        repeating-linear-gradient(to bottom, transparent 0 3px, rgba(255, 94, 19, 0.06) 4px),
        radial-gradient(circle at 50% 45%, rgba(255, 74, 9, 0.16), transparent 28%),
        #020202;
    opacity: 1;
    visibility: visible;
    transition: opacity 480ms ease, visibility 480ms ease;
}

#boot-sequence.boot-complete {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
}

.boot-frame {
    width: min(680px, calc(100vw - 36px));
    padding: 38px;
    border: 1px solid rgba(255, 103, 26, 0.42);
    clip-path: polygon(0 0, calc(100% - 26px) 0, 100% 26px, 100% 100%, 0 100%);
    background: rgba(5, 5, 5, 0.94);
    box-shadow: 0 0 90px rgba(255, 61, 5, 0.12);
}

.boot-emblem {
    display: flex;
    align-items: center;
    gap: 17px;
    margin-bottom: 30px;
    color: #f5eee6;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.boot-emblem svg {
    width: 58px;
    height: 58px;
}

.boot-line {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 111, 34, 0.10);
    color: #706a63;
    font-family: Consolas, monospace;
    font-size: 10px;
    letter-spacing: 0.13em;
    opacity: 0;
    transform: translateX(-8px);
    animation: boot-line-in 260ms ease forwards;
}

.boot-line strong {
    color: #00dff5;
    font-weight: 700;
}

.boot-line:nth-child(2) { animation-delay: 150ms; }
.boot-line:nth-child(3) { animation-delay: 360ms; }
.boot-line:nth-child(4) { animation-delay: 590ms; }
.boot-line:nth-child(5) { animation-delay: 820ms; }
.boot-line:nth-child(6) { animation-delay: 1050ms; }

.boot-progress {
    position: relative;
    height: 3px;
    margin-top: 28px;
    overflow: hidden;
    background: #17100c;
}

.boot-progress::after {
    content: "";
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: linear-gradient(90deg, #ff4308, #ff9b43, #00dff5);
    animation: boot-progress 1.55s cubic-bezier(.2, .7, .1, 1) forwards;
}

.brand-mark {
    cursor: pointer;
    user-select: none;
}

.brand-mark svg {
    width: 18px;
    height: 18px;
    margin-right: 8px;
    vertical-align: middle;
}

.mode-deck {
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
    min-height: 42px;
    margin: 0 0 10px !important;
    padding: 6px 10px !important;
    border: 0 !important;
    border-top: 1px solid rgba(255, 101, 27, 0.14) !important;
    border-bottom: 1px solid rgba(255, 101, 27, 0.22) !important;
    background: linear-gradient(90deg, rgba(255, 78, 10, 0.045), transparent 45%) !important;
}

.mode-deck > .block,
.mode-deck > div {
    min-height: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}

.matrix-label-block {
    flex: 0 0 auto !important;
    width: auto !important;
}

.matrix-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #80766d;
    font-family: Consolas, monospace;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    white-space: nowrap;
}

.matrix-label::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #ff661a;
    box-shadow: 0 0 10px rgba(255, 102, 26, 0.72);
}

#visual-mode {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}

#visual-mode .wrap {
    flex-wrap: nowrap !important;
    gap: 2px !important;
}

#visual-mode input[type="radio"] {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    margin: -1px !important;
    padding: 0 !important;
    overflow: hidden !important;
    clip: rect(0 0 0 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}

#visual-mode label {
    flex: 0 0 auto;
    min-width: 0;
    padding: 6px 11px !important;
    border: 0 !important;
    border-bottom: 1px solid transparent !important;
    color: #69625c !important;
    background: transparent !important;
    transition: color 160ms ease, border-color 160ms ease, background 160ms ease !important;
}

#visual-mode label:has(input:checked) {
    border-bottom-color: #ff6a1a !important;
    color: #f0e8e0 !important;
    background: rgba(255, 82, 11, 0.055) !important;
    box-shadow: none !important;
}

#visual-mode label:has(input:focus-visible) {
    outline: 1px solid rgba(0, 220, 245, 0.65) !important;
    outline-offset: 2px;
}

#interface-audio-toggle {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    margin-left: auto !important;
    align-self: center;
    padding: 3px 0 3px 13px !important;
    border-left: 1px solid rgba(0, 220, 245, 0.20) !important;
}

#interface-audio-toggle label {
    color: #647576 !important;
    font-size: 8px !important;
    letter-spacing: 0.12em !important;
    white-space: nowrap;
}

.voiceprint-shell {
    position: relative;
    height: 116px;
    margin-bottom: 12px;
    overflow: hidden;
    border: 1px solid rgba(0, 220, 244, 0.19);
    background:
        linear-gradient(90deg, rgba(0, 219, 243, 0.04), transparent),
        #030606;
}

#voiceprint-canvas,
#reactor-canvas {
    width: 100%;
    height: 100%;
    display: block;
}

.voiceprint-label {
    position: absolute;
    left: 10px;
    top: 9px;
    color: #00dff5;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.16em;
}

.voiceprint-readout {
    position: absolute;
    right: 10px;
    bottom: 8px;
    color: #668689;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.10em;
}

#synthesis-reactor {
    position: relative;
    height: 250px;
    margin-bottom: 13px;
    overflow: hidden;
    border: 1px solid rgba(255, 98, 24, 0.21);
    background:
        radial-gradient(circle at 50% 50%, rgba(255, 74, 9, 0.13), transparent 29%),
        #030303;
}

.reactor-core {
    position: absolute;
    left: 50%;
    top: 50%;
    display: grid;
    place-items: center;
    width: 98px;
    height: 98px;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(255, 111, 32, 0.48);
    border-radius: 50%;
    color: #ff8a3d;
    font-family: Consolas, monospace;
    font-size: 23px;
    letter-spacing: -0.06em;
    background: rgba(9, 4, 2, 0.72);
    box-shadow: 0 0 35px rgba(255, 72, 7, 0.13), inset 0 0 22px rgba(255, 72, 7, 0.09);
}

.reactor-core::before,
.reactor-core::after {
    content: "";
    position: absolute;
    border: 1px solid rgba(255, 103, 25, 0.18);
    border-radius: 50%;
}

.reactor-core::before { inset: -24px; }
.reactor-core::after { inset: -48px; border-style: dashed; }

.reactor-state {
    position: absolute;
    left: 12px;
    top: 10px;
    color: #6f6861;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.13em;
}

.reactor-log {
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: 10px;
    display: flex;
    justify-content: space-between;
    color: #577678;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.10em;
}

#synthesis-reactor.is-synthesizing .reactor-core {
    animation: reactor-pulse 900ms ease-in-out infinite;
}

#synthesis-reactor.is-ready .reactor-core {
    border-color: #00dff5;
    color: #00dff5;
    box-shadow: 0 0 38px rgba(0, 223, 245, 0.20);
}

.pipeline-stages {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 3px;
    margin: 0 0 13px;
}

.pipeline-stage {
    position: relative;
    min-width: 0;
    padding: 9px 6px 8px;
    border-top: 1px solid rgba(255, 100, 25, 0.14);
    color: #4d4843;
    font-family: Consolas, monospace;
    font-size: 7px;
    letter-spacing: 0.08em;
    text-align: center;
}

.pipeline-stage::before {
    content: "";
    position: absolute;
    left: 0;
    top: -2px;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #3b312b;
}

.pipeline-stage.active {
    color: #ff8b43;
    border-top-color: #ff5b15;
}

.pipeline-stage.active::before {
    background: #ff5b15;
    box-shadow: 0 0 10px #ff5b15;
}

.pipeline-stage.complete {
    color: #65c6d0;
    border-top-color: #00dff5;
}

.pipeline-stage.complete::before {
    background: #00dff5;
    box-shadow: 0 0 9px #00dff5;
}

.cover-art {
    max-width: 280px !important;
    margin: 12px auto 0 !important;
    border: 1px solid rgba(255, 100, 25, 0.20) !important;
    clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 0 100%);
}

.cartridge-array {
    margin: 12px 0;
}

.cartridge-summary {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 12px;
    border: 1px solid rgba(0, 220, 245, 0.14);
    color: #628588;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.09em;
    background: rgba(0, 210, 235, 0.035);
}

.cartridge-summary strong {
    color: #00dff5;
}

.cartridge-grid {
    display: grid;
    gap: 6px;
    margin-top: 7px;
}

.data-cartridge {
    position: relative;
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr) auto;
    align-items: center;
    gap: 10px;
    min-height: 64px;
    padding: 10px 24px 10px 10px;
    overflow: hidden;
    border: 1px solid rgba(255, 101, 25, 0.17);
    border-left: 2px solid #ff5b16;
    color: #d5cec5;
    background: linear-gradient(90deg, rgba(255, 78, 9, 0.07), transparent 34%), #050505;
}

.cartridge-index {
    color: #ff6b20;
    font-family: Consolas, monospace;
    font-size: 18px;
    text-align: center;
}

.cartridge-name {
    overflow: hidden;
    font-size: 12px;
    font-weight: 700;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cartridge-meta,
.cartridge-state {
    color: #6e6760;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.08em;
}

.cartridge-state {
    color: #a35e36;
    text-align: right;
}

.cartridge-signal {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: #4c1b0b;
}

.data-cartridge[data-state="running"] {
    border-color: rgba(255, 106, 26, 0.48);
    animation: cartridge-working 1.2s ease-in-out infinite;
}

.data-cartridge[data-state="running"] .cartridge-signal {
    background: #ff5b15;
    box-shadow: 0 0 15px #ff5b15;
}

.data-cartridge[data-state="complete"] {
    border-left-color: #00dff5;
}

.data-cartridge[data-state="complete"] .cartridge-state {
    color: #00dff5;
}

.data-cartridge[data-state="complete"] .cartridge-signal {
    background: #00dff5;
    box-shadow: 0 0 12px #00dff5;
}

.data-cartridge[data-state="failed"] {
    border-left-color: #ff254e;
}

.data-cartridge[data-state="failed"] .cartridge-state,
.data-cartridge[data-state="failed"] .cartridge-signal {
    color: #ff254e;
    background: #ff254e;
}

.array-standby {
    padding: 18px;
    border: 1px dashed rgba(255, 102, 25, 0.22);
    color: #665f58;
    font-family: Consolas, monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-align: center;
}

#command-strip {
    position: fixed;
    z-index: 800;
    left: 50%;
    bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    width: min(1180px, calc(100vw - 28px));
    min-height: 38px;
    padding: 7px 12px;
    transform: translateX(-50%);
    border: 1px solid rgba(255, 101, 25, 0.27);
    color: #77716a;
    font-family: Consolas, monospace;
    font-size: 8px;
    letter-spacing: 0.09em;
    background: rgba(3, 3, 3, 0.92);
    box-shadow: 0 14px 45px rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(16px);
}

.command-brand {
    color: #ff6a1a;
    font-weight: 800;
}

.command-status {
    color: #00dff5;
}

.command-strip-divider {
    width: 1px;
    align-self: stretch;
    background: rgba(255, 104, 29, 0.16);
}

.glitching {
    animation: signal-glitch 380ms steps(2, end);
}

.mode-polar.hero-shell {
    border-color: rgba(0, 223, 245, 0.42);
    background: linear-gradient(90deg, #020306 0%, #070710 50%, rgba(31, 4, 55, 0.78) 100%);
}

.mode-polar .noir-sun {
    background: repeating-linear-gradient(to bottom, #00dff5 0 11px, #31236e 11px 15px, transparent 15px 20px), linear-gradient(#93f7ff, #6c35d9);
    box-shadow: 0 0 85px rgba(0, 223, 245, 0.34), 0 0 180px rgba(110, 53, 217, 0.18);
}

.mode-polar.hero-shell h1 span {
    background: linear-gradient(92deg, #00dff5, #86f5ff 62%, #b38aff);
    background-clip: text;
    -webkit-background-clip: text;
}

.mode-polar.studio-card {
    border-color: rgba(0, 220, 245, 0.22) !important;
    border-left-color: #00dff5 !important;
    background: linear-gradient(125deg, rgba(0, 220, 245, 0.05), transparent 36%), rgba(5, 6, 10, 0.95) !important;
}

.mode-polar.studio-card > .block:first-child,
.mode-polar.studio-card > div:first-child {
    background: linear-gradient(100deg, rgba(0, 151, 180, 0.34), rgba(66, 34, 122, 0.42)) !important;
}

.mode-polar.studio-card .section-eyebrow,
.mode-polar.mode-deck .section-eyebrow {
    color: #00dff5;
}

.mode-polar #generate-btn,
.mode-polar #queue-btn {
    border-color: #a9f8ff !important;
    color: #02070a !important;
    background: linear-gradient(95deg, #00cce6, #80efff 62%, #b38aff) !important;
    box-shadow: 0 0 32px rgba(0, 219, 245, 0.22) !important;
}

.mode-polar.tabs-shell > .tab-nav button.selected {
    border-color: #00dff5 !important;
    color: #02090b !important;
    background: linear-gradient(90deg, #00bfd8, #83f1ff) !important;
    box-shadow: 0 0 22px rgba(0, 223, 245, 0.18) !important;
}

.mode-polar #visual-mode label:has(input:checked) {
    border-bottom-color: #00dff5 !important;
    color: #bdfaff !important;
    background: rgba(0, 223, 245, 0.055) !important;
    box-shadow: none !important;
}

.mode-polar .matrix-label::before {
    background: #00dff5;
    box-shadow: 0 0 10px rgba(0, 223, 245, 0.75);
}

.mode-polar#command-strip,
.mode-polar.mode-deck {
    border-color: rgba(0, 223, 245, 0.25) !important;
}

.mode-void.hero-shell {
    filter: saturate(0.12);
    border-color: rgba(255, 255, 255, 0.28);
    background: linear-gradient(90deg, #020202, #0b0b0b 62%, #160206);
}

.mode-void .noir-sun {
    background: repeating-linear-gradient(to bottom, #e8e8e8 0 11px, #3b3b3b 11px 15px, transparent 15px 20px), linear-gradient(#fff, #b5b5b5);
    box-shadow: 0 0 70px rgba(255, 255, 255, 0.18);
}

.mode-void.hero-shell h1 span {
    background: linear-gradient(92deg, #ffffff, #c3c3c3 64%, #ff3158);
    background-clip: text;
    -webkit-background-clip: text;
}

.mode-void.studio-card {
    border-color: rgba(255, 255, 255, 0.16) !important;
    border-left-color: #ff214f !important;
    background: linear-gradient(125deg, rgba(255, 32, 73, 0.035), transparent 36%), rgba(5, 5, 5, 0.96) !important;
}

.mode-void.studio-card > .block:first-child,
.mode-void.studio-card > div:first-child {
    background: linear-gradient(100deg, rgba(255, 255, 255, 0.10), rgba(122, 5, 31, 0.28)) !important;
}

.mode-void.studio-card .section-eyebrow {
    color: #ff3158;
}

.mode-void #generate-btn,
.mode-void #queue-btn {
    border-color: #ff6b82 !important;
    color: #fff !important;
    background: linear-gradient(95deg, #8e0724, #ff214f) !important;
    box-shadow: 0 0 30px rgba(255, 33, 79, 0.20) !important;
}

.mode-void.tabs-shell > .tab-nav button.selected {
    border-color: #ff3158 !important;
    color: #ffffff !important;
    background: linear-gradient(90deg, #8e0724, #ff3158) !important;
    box-shadow: 0 0 22px rgba(255, 49, 88, 0.18) !important;
}

.mode-void #visual-mode label:has(input:checked) {
    border-bottom-color: #ff3158 !important;
    color: #ffffff !important;
    background: rgba(255, 49, 88, 0.055) !important;
    box-shadow: none !important;
}

.mode-void .matrix-label::before {
    background: #ff3158;
    box-shadow: 0 0 10px rgba(255, 49, 88, 0.75);
}

.mode-void#command-strip,
.mode-void.mode-deck {
    border-color: rgba(255, 49, 88, 0.28) !important;
}

@keyframes boot-line-in {
    to { opacity: 1; transform: translateX(0); }
}

@keyframes boot-progress {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

@keyframes reactor-pulse {
    0%, 100% { transform: translate(-50%, -50%) scale(0.94); box-shadow: 0 0 25px rgba(255, 72, 7, 0.13); }
    50% { transform: translate(-50%, -50%) scale(1.06); box-shadow: 0 0 55px rgba(255, 72, 7, 0.27); }
}

@keyframes cartridge-working {
    0%, 100% { background-color: #050505; }
    50% { background-color: #120804; }
}

@keyframes signal-glitch {
    0%, 100% { transform: translate(0); filter: none; }
    25% { transform: translate(-3px, 1px); filter: hue-rotate(34deg); }
    50% { transform: translate(3px, -1px); opacity: 0.72; }
    75% { transform: translate(-1px, 0); filter: contrast(1.8); }
}

@media (max-width: 900px) {
    .mode-deck {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 5px 8px !important;
    }

    #interface-audio-toggle {
        margin-left: auto !important;
    }

    #synthesis-reactor {
        height: 220px;
    }

    .pipeline-stages {
        grid-template-columns: 1fr;
    }

    .pipeline-stage {
        text-align: left;
    }

    .data-cartridge {
        grid-template-columns: 36px minmax(0, 1fr);
    }

    .cartridge-state {
        grid-column: 2;
        text-align: left;
    }

    #command-strip {
        gap: 8px;
        overflow: hidden;
        white-space: nowrap;
    }

    #command-strip .command-hide-mobile,
    #command-strip .command-strip-divider {
        display: none;
    }
}

@media (max-width: 520px) {
    .matrix-label-block {
        display: none !important;
    }

    #visual-mode label {
        padding-inline: 8px !important;
    }

    #interface-audio-toggle {
        padding-left: 8px !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    .boot-line,
    .boot-progress::after,
    #synthesis-reactor.is-synthesizing .reactor-core,
    .data-cartridge[data-state="running"],
    .glitching {
        animation: none !important;
    }
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

APP_JS = r"""
(() => {
    const startDeck = () => {
        if (window.__cbxDeckStarted || !document.querySelector(".hero-shell")) {
            if (!window.__cbxDeckStarted) setTimeout(startDeck, 120);
            return;
        }
        window.__cbxDeckStarted = true;

        const qs = (selector, root = document) => root.querySelector(selector);
        const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));
        const deckState = {
            mode: "ember",
            synthesizing: false,
            percent: 0,
            audioContext: null,
            stageTimers: [],
        };

        const boot = qs("#boot-sequence");
        const replayBoot = () => {
            if (!boot) return;
            boot.setAttribute("aria-hidden", "false");
            boot.classList.remove("boot-complete");
            void boot.offsetWidth;
            const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 120 : 1900;
            window.setTimeout(() => {
                boot.classList.add("boot-complete");
                boot.setAttribute("aria-hidden", "true");
            }, delay);
        };
        replayBoot();

        const brand = qs(".brand-mark");
        if (brand) {
            brand.setAttribute("role", "button");
            brand.setAttribute("tabindex", "0");
            brand.setAttribute("title", "Replay system boot");
            brand.addEventListener("click", replayBoot);
            brand.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") replayBoot();
            });
        }

        const matrixTargets = () => qsa(
            ".hero-shell, .studio-card, .tabs-shell, .mode-deck, #command-strip"
        );
        const rootContainer = qs(".gradio-container");
        const matrixBackgrounds = {
            ember: "linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px),radial-gradient(circle at 82% 2%,rgba(255,78,14,.20),transparent 27%),radial-gradient(circle at 4% 48%,rgba(0,213,255,.07),transparent 25%),linear-gradient(135deg,#030303,#080706 52%,#050505)",
            polar: "linear-gradient(rgba(0,223,245,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,223,245,.025) 1px,transparent 1px),radial-gradient(circle at 82% 2%,rgba(98,54,210,.22),transparent 29%),radial-gradient(circle at 3% 48%,rgba(0,223,245,.12),transparent 27%),linear-gradient(135deg,#020306,#060712 55%,#05030a)",
            void: "linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px),radial-gradient(circle at 86% 3%,rgba(255,31,76,.12),transparent 24%),linear-gradient(135deg,#020202,#080808 62%,#070203)",
        };
        const applyMode = (mode) => {
            deckState.mode = mode;
            matrixTargets().forEach((element) => {
                element.classList.remove("mode-ember", "mode-polar", "mode-void");
                element.classList.add(`mode-${mode}`);
            });
            if (rootContainer) {
                rootContainer.style.setProperty(
                    "background",
                    matrixBackgrounds[mode],
                    "important"
                );
                rootContainer.style.setProperty("background-size", "44px 44px,44px 44px,auto,auto,auto", "important");
            }
            const matrixReadout = qs("#matrix-readout");
            if (matrixReadout) matrixReadout.textContent = mode.toUpperCase();
        };
        applyMode("ember");

        const readComponentValue = (id) => {
            const component = qs(`#${id}`);
            if (!component) return "--";
            const input = qs("input", component) || qs("textarea", component);
            return (input && input.value) || "--";
        };
        const compactModel = (value) => value.includes("V3") ? "V3" : value.includes("Turbo") ? "TURBO" : value;
        const compactFormat = (value) => value.startsWith("MP3") ? "MP3 / 160K" : value.startsWith("WAV") ? "WAV / PCM" : value;
        const updateCommandStrip = () => {
            const model = qs("#cmd-model");
            const language = qs("#cmd-language");
            const format = qs("#cmd-format");
            const modelValue = compactModel(readComponentValue("model-choice"));
            const languageValue = readComponentValue("language-choice").toUpperCase();
            const formatValue = compactFormat(readComponentValue("format-choice"));
            if (model && model.textContent !== modelValue) model.textContent = modelValue;
            if (language && language.textContent !== languageValue) language.textContent = languageValue;
            if (format && format.textContent !== formatValue) format.textContent = formatValue;
        };
        updateCommandStrip();

        const tone = (frequency, duration = 0.045, type = "square", volume = 0.018) => {
            const toggle = qs("#interface-audio-toggle input[type='checkbox']");
            if (!toggle || !toggle.checked) return;
            try {
                deckState.audioContext = deckState.audioContext || new (window.AudioContext || window.webkitAudioContext)();
                const context = deckState.audioContext;
                const oscillator = context.createOscillator();
                const gain = context.createGain();
                oscillator.type = type;
                oscillator.frequency.setValueAtTime(frequency, context.currentTime);
                gain.gain.setValueAtTime(volume, context.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + duration);
                oscillator.connect(gain).connect(context.destination);
                oscillator.start();
                oscillator.stop(context.currentTime + duration);
            } catch (_) {}
        };
        const readyChime = () => {
            tone(330, 0.08, "sine", 0.025);
            setTimeout(() => tone(660, 0.12, "sine", 0.021), 90);
        };

        const palette = () => {
            if (deckState.mode === "polar") return {signal: "#00e7ff", accent: "#9d75ff", dim: "rgba(0,223,245,.12)"};
            if (deckState.mode === "void") return {signal: "#ff3158", accent: "#eeeeee", dim: "rgba(255,49,88,.12)"};
            return {signal: "#00dff5", accent: "#ff6518", dim: "rgba(255,101,24,.12)"};
        };

        const animateCanvas = (canvasId, reactor = false) => {
            const canvas = qs(`#${canvasId}`);
            if (!canvas) return;
            const context = canvas.getContext("2d");
            let phase = Math.random() * 20;
            let last = 0;
            const resize = () => {
                const ratio = Math.min(window.devicePixelRatio || 1, 2);
                canvas.width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
                canvas.height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
                context.setTransform(ratio, 0, 0, ratio, 0, 0);
            };
            resize();
            new ResizeObserver(resize).observe(canvas);
            const draw = (time) => {
                requestAnimationFrame(draw);
                if (time - last < 42) return;
                last = time;
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                const colors = palette();
                const energy = deckState.synthesizing ? 1 : 0.34;
                context.clearRect(0, 0, width, height);
                context.strokeStyle = colors.dim;
                context.lineWidth = 1;
                for (let x = 0; x < width; x += 26) {
                    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
                }
                for (let y = 0; y < height; y += 24) {
                    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
                }
                if (reactor) {
                    const cx = width / 2;
                    const cy = height / 2;
                    for (let ring = 0; ring < 4; ring += 1) {
                        const radius = 33 + ring * 25 + Math.sin(phase * 0.7 + ring) * 4 * energy;
                        context.beginPath();
                        context.arc(cx, cy, radius, 0, Math.PI * 2);
                        context.strokeStyle = ring % 2 ? colors.signal : colors.accent;
                        context.globalAlpha = 0.15 + energy * 0.16;
                        context.stroke();
                    }
                    context.globalAlpha = 1;
                } else {
                    const bars = Math.max(26, Math.floor(width / 8));
                    for (let index = 0; index < bars; index += 1) {
                        const x = (index + 0.5) * width / bars;
                        const wave = Math.sin(index * 0.71 + phase) * 0.42 + Math.sin(index * 0.21 - phase * 1.3) * 0.28;
                        const amplitude = (0.22 + Math.abs(wave) * 0.72) * height * energy;
                        context.beginPath();
                        context.moveTo(x, height / 2 - amplitude / 2);
                        context.lineTo(x, height / 2 + amplitude / 2);
                        context.strokeStyle = index % 7 === 0 ? colors.accent : colors.signal;
                        context.globalAlpha = 0.45 + energy * 0.45;
                        context.stroke();
                    }
                    context.globalAlpha = 1;
                }
                phase += 0.055 + energy * 0.09;
            };
            requestAnimationFrame(draw);
        };
        animateCanvas("voiceprint-canvas", false);
        animateCanvas("reactor-canvas", true);

        const stages = qsa(".pipeline-stage");
        const reactor = qs("#synthesis-reactor");
        const reactorPercent = qs("#reactor-percent");
        const reactorStatus = qs("#reactor-status");
        const reactorLog = qs("#reactor-log-line");
        const clearStageTimers = () => {
            deckState.stageTimers.forEach(clearTimeout);
            deckState.stageTimers = [];
        };
        const setStage = (index) => {
            stages.forEach((stage, stageIndex) => {
                stage.classList.toggle("complete", stageIndex < index);
                stage.classList.toggle("active", stageIndex === index);
            });
            const labels = ["PARSING TRANSMISSION", "ENCODING IDENTITY", "SYNTHESIZING SPEECH", "MASTERING ARTIFACT", "ARCHIVING OUTPUT"];
            if (reactorStatus) reactorStatus.textContent = labels[index] || "SYNTHESIS ACTIVE";
            if (reactorLog) reactorLog.textContent = `PHASE ${String(index + 1).padStart(2, "0")} // ${labels[index] || "PROCESSING"}`;
        };
        const beginSynthesis = () => {
            clearStageTimers();
            deckState.synthesizing = true;
            deckState.percent = 1;
            if (reactor) {
                reactor.classList.remove("is-ready");
                reactor.classList.add("is-synthesizing");
            }
            const heroTitle = qs(".hero-shell h1");
            if (heroTitle) {
                heroTitle.classList.add("glitching");
                setTimeout(() => heroTitle.classList.remove("glitching"), 420);
            }
            setStage(0);
            [650, 1800, 3800, 6100].forEach((delay, index) => {
                deckState.stageTimers.push(setTimeout(() => setStage(index + 1), delay));
            });
            const percentTimer = setInterval(() => {
                if (!deckState.synthesizing) return clearInterval(percentTimer);
                deckState.percent = Math.min(93, deckState.percent + Math.max(1, Math.floor((96 - deckState.percent) / 15)));
                if (reactorPercent) reactorPercent.textContent = String(deckState.percent).padStart(3, "0");
            }, 430);
            tone(92, 0.09, "sawtooth", 0.025);
            setTimeout(() => tone(138, 0.10, "square", 0.018), 95);
            const commandStatus = qs("#cmd-status");
            if (commandStatus) commandStatus.textContent = "SYNTHESIS ACTIVE";
        };
        const finishSynthesis = () => {
            if (!deckState.synthesizing) return;
            clearStageTimers();
            deckState.synthesizing = false;
            deckState.percent = 100;
            stages.forEach((stage) => {
                stage.classList.remove("active");
                stage.classList.add("complete");
            });
            if (reactor) {
                reactor.classList.remove("is-synthesizing");
                reactor.classList.add("is-ready");
            }
            if (reactorPercent) reactorPercent.textContent = "100";
            if (reactorStatus) reactorStatus.textContent = "ARTIFACT READY";
            if (reactorLog) reactorLog.textContent = "ARCHIVE LOCKED // AUDIO + COVER WRITTEN";
            const commandStatus = qs("#cmd-status");
            if (commandStatus) commandStatus.textContent = "ARTIFACT READY";
            readyChime();
        };

        const outputComponent = qs("#audio-output");
        if (outputComponent) {
            new MutationObserver(() => {
                const audio = qs("audio", outputComponent);
                const artifactLink = qs("a", outputComponent);
                if ((audio && (audio.currentSrc || audio.src)) || artifactLink) finishSynthesis();
            }).observe(outputComponent, {subtree: true, childList: true, attributes: true, attributeFilter: ["src"]});
        }

        const referenceComponent = qs("#reference-audio");
        const updateVoiceprint = () => {
            const readout = qs("#voiceprint-status");
            if (!readout || !referenceComponent) return;
            const audio = qs("audio", referenceComponent);
            const downloadLink = qs("a", referenceComponent);
            const hasSignal = (audio && (audio.currentSrc || audio.src)) || downloadLink;
            readout.textContent = hasSignal ? "VOICEPRINT LOCKED // IDENTITY SIGNAL ONLINE" : "AWAITING IDENTITY SIGNAL";
        };
        if (referenceComponent) {
            new MutationObserver(updateVoiceprint).observe(referenceComponent, {subtree: true, childList: true, attributes: true});
            updateVoiceprint();
        }

        document.addEventListener("change", (event) => {
            const target = event.target;
            if (target.closest && target.closest("#visual-mode") && target.value) {
                applyMode(String(target.value).toLowerCase());
                tone(180, 0.05, "square", 0.018);
            }
            if (target.closest && ["model-choice", "language-choice", "format-choice"].some((id) => target.closest(`#${id}`))) {
                updateCommandStrip();
                const heroTitle = qs(".hero-shell h1");
                if (heroTitle) {
                    heroTitle.classList.add("glitching");
                    setTimeout(() => heroTitle.classList.remove("glitching"), 420);
                }
            }
        });

        document.addEventListener("click", (event) => {
            const button = event.target.closest && event.target.closest("button");
            if (!button) return;
            if (button.closest("#generate-btn")) {
                beginSynthesis();
                return;
            }
            if (button.closest("#queue-btn")) {
                const cartridges = qsa(".data-cartridge");
                cartridges.forEach((cartridge) => {
                    cartridge.dataset.state = "queued";
                    const state = qs(".cartridge-state", cartridge);
                    if (state) state.textContent = "QUEUED";
                });
                cartridges.forEach((cartridge, index) => {
                    setTimeout(() => {
                        cartridges.forEach((item) => {
                            if (item.dataset.state === "running") item.dataset.state = "queued";
                        });
                        cartridge.dataset.state = "running";
                        const state = qs(".cartridge-state", cartridge);
                        if (state) state.textContent = "SYNTHESIZING";
                    }, index * 1400);
                });
                tone(110, 0.08, "sawtooth", 0.022);
                return;
            }
            if (!button.closest("audio")) tone(155, 0.032, "square", 0.012);
        });

        new MutationObserver(() => {
            updateCommandStrip();
            if (deckState.synthesizing && qs("#audio-output a")) finishSynthesis();
        }).observe(document.body, {subtree: true, childList: true});
    };
    startDeck();
})()
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(model_choice):
    model_key = "v3" if model_choice == MODEL_V3 else "turbo"
    if model_key not in MODEL_CACHE:
        if model_key == "v3":
            print(f"Loading Chatterbox Multilingual V3 on {DEVICE}...")
            model = ChatterboxMultilingualTTS.from_pretrained(
                DEVICE,
                t3_model="v3",
            )
        else:
            print(f"Loading Chatterbox-Turbo on {DEVICE}...")
            model = ChatterboxTurboTTS.from_pretrained(DEVICE)
        MODEL_CACHE[model_key] = model
    return MODEL_CACHE[model_key]


def model_ui_state(model_choice):
    if model_choice == MODEL_V3:
        message = (
            "**V3 selected:** best naturalness, voice similarity, and stability. "
            "Turbo-only bracketed tags are hidden and removed before synthesis."
        )
        return message, gr.update(visible=False)

    message = (
        "**Turbo selected:** faster generation with native sound and style tags. "
        "Choose V3 when absolute voice quality matters more than effects."
    )
    return message, gr.update(visible=True)


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long documents into natural, model-sized sections."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        pieces = textwrap.wrap(
            sentence,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [sentence]

        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate

    if current:
        chunks.append(current)

    return chunks


def parse_document(text: str) -> list[tuple[str, str | float]]:
    """Turn a document into model-sized speech and explicit silence segments."""
    segments: list[tuple[str, str | float]] = []
    cursor = 0

    for match in PAUSE_TAG_PATTERN.finditer(text):
        for chunk in split_text(text[cursor:match.start()]):
            segments.append(("speech", chunk))

        raw_duration = match.group(1)
        unit = (match.group(2) or "s").lower()
        seconds = (
            float(raw_duration)
            if raw_duration is not None
            else DEFAULT_EXPLICIT_PAUSE_SECONDS
        )
        if unit == "ms":
            seconds /= 1000
        seconds = min(max(seconds, 0.0), MAX_EXPLICIT_PAUSE_SECONDS)
        if seconds > 0:
            segments.append(("pause", seconds))
        cursor = match.end()

    for chunk in split_text(text[cursor:]):
        segments.append(("speech", chunk))

    return segments


def read_text_file(file_path) -> str:
    path = Path(file_path)
    raw = path.read_bytes()
    document = None

    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            document = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if document is None:
        raise gr.Error("This text file uses an unsupported character encoding.")

    if not any(kind == "speech" for kind, _ in parse_document(document)):
        raise gr.Error("The uploaded text file is empty.")
    return document


def normalize_file_paths(file_paths) -> list[Path]:
    if not file_paths:
        return []
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    return [Path(file_path) for file_path in file_paths]


def render_queue_cartridges(records, summary: str) -> str:
    cards = []
    for index, record in enumerate(records, start=1):
        state = record.get("state", "queued")
        filename = escape(str(record["name"]))
        detail = escape(str(record.get("detail", "AWAITING SYNTHESIS")))
        sections = int(record.get("sections", 0))
        characters = int(record.get("characters", 0))
        cards.append(
            f"""
            <article class="data-cartridge" data-state="{state}">
                <div class="cartridge-index">{index:02d}</div>
                <div class="cartridge-body">
                    <div class="cartridge-name">{filename}</div>
                    <div class="cartridge-meta">
                        {characters:,} CHR &nbsp;/&nbsp; {sections:,} SEG
                    </div>
                </div>
                <div class="cartridge-state">{detail}</div>
                <div class="cartridge-signal" aria-hidden="true"></div>
            </article>
            """
        )

    return (
        '<section class="cartridge-array">'
        f'<header class="cartridge-summary"><span>{escape(summary)}</span>'
        f'<strong>{len(records):02d} DATACARTRIDGES</strong></header>'
        f'<div class="cartridge-grid">{"".join(cards)}</div>'
        '</section>'
    )


def load_text_files(file_paths):
    paths = normalize_file_paths(file_paths)
    if not paths:
        return gr.update(), (
            '<div class="array-standby">ARRAY EMPTY // LOAD .TXT ARTIFACTS</div>'
        )

    documents = []
    records = []
    total_characters = 0
    total_sections = 0
    total_pauses = 0

    for path in paths:
        document = read_text_file(path)
        documents.append(document)
        segments = parse_document(document)
        section_count = sum(kind == "speech" for kind, _ in segments)
        pause_count = sum(kind == "pause" for kind, _ in segments)
        total_characters += len(document)
        total_sections += section_count
        total_pauses += pause_count
        records.append(
            {
                "name": path.name,
                "characters": len(document),
                "sections": section_count,
                "state": "queued",
                "detail": "QUEUED",
            }
        )

    summary = (
        f"{total_characters:,} CHR // {total_sections:,} SEG"
        + (f" // {total_pauses:,} PAUSE" if total_pauses else "")
    )
    return documents[0], render_queue_cartridges(records, summary)


def make_output_path(
        output_stem: str | None,
        model_choice: str,
        output_format: str,
) -> Path:
    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        output_stem or "chatterbox",
    ).strip("._-")
    safe_stem = (safe_stem or "chatterbox")[:80]
    model_slug = "v3" if model_choice == MODEL_V3 else "turbo"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid4().hex[:8]
    extension = ".mp3" if output_format == FORMAT_MP3 else ".wav"
    return OUTPUT_DIR / (
        f"{safe_stem}-{model_slug}-{timestamp}-{unique_id}{extension}"
    )


def _cover_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/bahnschrift.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_cover_art(
        output_path: Path,
        model_choice: str,
        output_stem: str | None,
) -> Path | None:
    """Create original cover art for every audio artifact."""
    cover_path = output_path.with_suffix(".png")
    try:
        size = 1200
        image = Image.new("RGB", (size, size), "#050505")
        draw = ImageDraw.Draw(image)

        for y in range(size):
            ember = max(0.0, 1.0 - abs(y - 250) / 620)
            draw.line(
                (0, y, size, y),
                fill=(
                    int(5 + 30 * ember),
                    int(5 + 7 * ember),
                    int(5 + 2 * ember),
                ),
            )

        for position in range(0, size + 1, 72):
            draw.line((position, 0, position, size), fill="#17100c", width=2)
            draw.line((0, position, size, position), fill="#17100c", width=2)

        sun_box = (650, 110, 1100, 560)
        draw.ellipse(sun_box, fill="#ff6516")
        for stripe_y in range(135, 555, 28):
            draw.rectangle((630, stripe_y, 1120, stripe_y + 9), fill="#541203")

        draw.line((75, 660, 1125, 660), fill="#ff641a", width=3)
        draw.line((75, 672, 570, 672), fill="#00dff5", width=3)
        draw.polygon(
            [(56, 46), (1065, 46), (1144, 125), (1144, 1148), (56, 1148)],
            outline="#7c2a0d",
            width=3,
        )

        title_font = _cover_font(168, bold=True)
        machine_font = _cover_font(145, bold=True)
        mono_font = _cover_font(30)
        small_font = _cover_font(24)

        draw.text((76, 710), "VOICE", font=title_font, fill="#f4ede4")
        draw.text((76, 855), "MACHINE", font=machine_font, fill="#f05b1b")

        artifact_name = re.sub(r"\s+", " ", output_stem or "UNTITLED").strip()
        artifact_name = artifact_name[:42].upper() or "UNTITLED"
        model_label = "MULTILINGUAL V3" if model_choice == MODEL_V3 else "TURBO"
        artifact_id = output_path.stem[-17:].upper()
        draw.text((80, 85), "CBX // SYNTHETIC VOICE DIVISION", font=mono_font, fill="#ff8a42")
        draw.text((80, 1080), artifact_name, font=mono_font, fill="#ded8cf")
        draw.text(
            (80, 1121),
            f"MODEL // {model_label}    ARTIFACT // {artifact_id}",
            font=small_font,
            fill="#69cfda",
        )

        image.save(cover_path, "PNG", optimize=True)
        return cover_path
    except Exception as exc:
        print(f"Cover art generation skipped: {exc}")
        cover_path.unlink(missing_ok=True)
        return None


def generate(
        model_choice,
        language_id,
        text,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        exaggeration,
        cfg_weight,
        output_format=FORMAT_MP3,
        output_stem=None,
        progress=gr.Progress()
):
    model = load_model(model_choice)
    use_v3 = model_choice == MODEL_V3

    if seed_num != 0:
        set_seed(int(seed_num))

    segments = parse_document(text)
    if not any(kind == "speech" for kind, _ in segments):
        raise gr.Error("Enter some text or upload a `.txt` file first.")

    processed_segments: list[tuple[str, str | float]] = []
    for kind, value in segments:
        if kind == "pause":
            processed_segments.append((kind, value))
            continue

        chunk = str(value)
        if use_v3:
            for tag in TURBO_TAGS:
                chunk = chunk.replace(tag, "")
            chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            processed_segments.append(("speech", chunk))

    if not any(kind == "speech" for kind, _ in processed_segments):
        raise gr.Error("No readable text remains after removing Turbo-only tags.")

    output_path = make_output_path(
        output_stem,
        model_choice,
        output_format,
    )
    working_path = (
        output_path.with_suffix(".processing.wav")
        if output_format == FORMAT_MP3
        else output_path
    )

    chunk_pause = np.zeros(
        int(model.sr * CHUNK_PAUSE_SECONDS),
        dtype=np.float32,
    )
    speech_total = sum(
        kind == "speech"
        for kind, _ in processed_segments
    )
    speech_index = 0
    cover_path = None

    try:
        with sf.SoundFile(
            str(working_path),
            mode="w",
            samplerate=model.sr,
            channels=1,
            subtype="PCM_16",
        ) as combined_audio:
            for index, (kind, value) in enumerate(processed_segments):
                if kind == "pause":
                    seconds = float(value)
                    progress(
                        index / len(processed_segments),
                        desc=f"Adding {seconds:g}-second pause",
                    )
                    combined_audio.write(
                        np.zeros(int(model.sr * seconds), dtype=np.float32)
                    )
                    continue

                speech_index += 1
                chunk = str(value)
                progress(
                    index / len(processed_segments),
                    desc=f"Reading section {speech_index} of {speech_total}",
                )
                if use_v3:
                    wav = model.generate(
                        chunk,
                        language_id=language_id,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        min_p=min_p,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                else:
                    wav = model.generate(
                        chunk,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        min_p=min_p,
                        top_p=top_p,
                        top_k=int(top_k),
                        repetition_penalty=repetition_penalty,
                        norm_loudness=norm_loudness,
                    )
                combined_audio.write(
                    wav.detach().cpu().squeeze().float().numpy()
                )
                if (
                    index < len(processed_segments) - 1
                    and processed_segments[index + 1][0] == "speech"
                ):
                    combined_audio.write(chunk_pause)

        progress(0.985, desc="Rendering artifact cover")
        cover_title = output_stem
        if not cover_title:
            title_text = PAUSE_TAG_PATTERN.sub(" ", text)
            for tag in TURBO_TAGS:
                title_text = title_text.replace(tag, " ")
            cover_title = " ".join(title_text.split()[:7]) or "UNTITLED"
        cover_path = create_cover_art(
            output_path,
            model_choice,
            cover_title,
        )

        if output_format == FORMAT_MP3:
            if not FFMPEG_EXE:
                raise RuntimeError(
                    "MP3 encoding requires FFmpeg, but it was not found."
                )
            progress(0.99, desc="Encoding maximum-quality MP3")
            ffmpeg_command = [
                FFMPEG_EXE,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(working_path),
            ]
            if cover_path:
                ffmpeg_command.extend(
                    [
                        "-i",
                        str(cover_path),
                        "-map",
                        "0:a",
                        "-map",
                        "1:v",
                    ]
                )
            ffmpeg_command.extend(
                [
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "160k",
                ]
            )
            if cover_path:
                ffmpeg_command.extend(
                    [
                        "-codec:v",
                        "png",
                        "-disposition:v",
                        "attached_pic",
                        "-id3v2_version",
                        "3",
                        "-metadata:s:v",
                        "title=CBX ARTIFACT COVER",
                        "-metadata:s:v",
                        "comment=Cover (front)",
                    ]
                )
            ffmpeg_command.extend(
                [
                    "-metadata",
                    f"title={cover_title}",
                    "-metadata",
                    "artist=CBX Voice Machine",
                    str(output_path),
                ]
            )
            subprocess.run(
                ffmpeg_command,
                check=True,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            working_path.unlink(missing_ok=True)
    except Exception:
        working_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        if cover_path:
            cover_path.unlink(missing_ok=True)
        raise

    progress(1, desc="Finished")
    return (
        str(output_path),
        gr.update(
            value=str(cover_path) if cover_path else None,
            visible=bool(cover_path),
        ),
    )


def process_file_queue(
        file_paths,
        model_choice,
        output_format,
        language_id,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        exaggeration,
        cfg_weight,
        progress=gr.Progress()
):
    paths = normalize_file_paths(file_paths)
    if not paths:
        raise gr.Error("Add at least one `.txt` file to the queue first.")

    completed = []
    failures = []
    records = []

    for file_index, path in enumerate(paths):
        record = {
            "name": path.name,
            "characters": 0,
            "sections": 0,
            "state": "running",
            "detail": "SYNTHESIZING",
        }
        try:
            document = read_text_file(path)
            segments = parse_document(document)
            record["characters"] = len(document)
            record["sections"] = sum(
                kind == "speech" for kind, _ in segments
            )

            def file_progress(value, desc=None):
                overall = (file_index + float(value)) / len(paths)
                progress(
                    overall,
                    desc=f"{path.name}: {desc or 'processing'}",
                )

            generate(
                model_choice,
                language_id,
                document,
                audio_prompt_path,
                temperature,
                seed_num,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
                norm_loudness,
                exaggeration,
                cfg_weight,
                output_format=output_format,
                output_stem=path.stem,
                progress=file_progress,
            )
            completed.append(path.name)
            record["state"] = "complete"
            record["detail"] = "ARTIFACT READY"
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            record["state"] = "failed"
            record["detail"] = "SYNTHESIS FAILED"
        records.append(record)

    progress(1, desc="Queue finished")
    if not completed:
        raise gr.Error("No queued files could be generated.")
    summary = (
        f"{len(completed):02d} COMPLETE // {len(failures):02d} FAILED // "
        f"ARCHIVE {OUTPUT_DIR.name.upper()}"
    )
    return render_queue_cartridges(records, summary)


with gr.Blocks(title="Chatterbox Studio") as demo:
    gr.HTML(
        """
        <div id="boot-sequence" aria-label="CBX system startup">
            <div class="boot-frame">
                <div class="boot-emblem">
                    <svg viewBox="0 0 64 64" aria-hidden="true">
                        <path d="M5 4h44l10 10v46H5z" fill="#050505" stroke="#ff671f"/>
                        <circle cx="32" cy="27" r="14" fill="#ff5d14"/>
                        <path d="M17 22h30M17 27h30M17 32h30" stroke="#541203" stroke-width="2"/>
                        <path d="M11 48h14l4-4 5 7 6-10 5 7h8" fill="none" stroke="#00dff5"/>
                    </svg>
                    CBX // VOICE MACHINE
                </div>
                <div class="boot-line"><span>LOCAL SYNTHESIS CORE</span><strong>ONLINE</strong></div>
                <div class="boot-line"><span>CUDA COMPUTE LINK</span><strong>STABLE</strong></div>
                <div class="boot-line"><span>IDENTITY ENCODER</span><strong>ARMED</strong></div>
                <div class="boot-line"><span>ARTIFACT ARCHIVE</span><strong>MOUNTED</strong></div>
                <div class="boot-line"><span>NETWORK EXPOSURE</span><strong>LOCAL ONLY</strong></div>
                <div class="boot-progress" aria-hidden="true"></div>
            </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["mode-deck"]):
        gr.HTML(
            '<div class="matrix-label">Visual matrix</div>',
            elem_classes=["matrix-label-block"],
        )
        visual_mode = gr.Radio(
            choices=["EMBER", "POLAR", "VOID"],
            value="EMBER",
            label="Visual matrix",
            show_label=False,
            container=False,
            elem_id="visual-mode",
            interactive=True,
        )
        interface_audio = gr.Checkbox(
            value=False,
            label="UI tones",
            show_label=False,
            container=False,
            elem_id="interface-audio-toggle",
        )

    gr.HTML(
        f"""
        <section class="hero-shell">
            <div class="hero-copy">
                <div class="brand-mark">
                    <svg viewBox="0 0 64 64" aria-hidden="true">
                        <path d="M5 4h44l10 10v46H5z" fill="#050505" stroke="currentColor"/>
                        <circle cx="32" cy="27" r="14" fill="currentColor" opacity=".78"/>
                        <path d="M17 22h30M17 27h30M17 32h30" stroke="#401006" stroke-width="2"/>
                        <path d="M11 48h14l4-4 5 7 6-10 5 7h8" fill="none" stroke="#00dff5"/>
                    </svg>
                    CBX // 49
                </div>
                <p class="eyebrow">Synthetic voice division · offline array</p>
                <h1>Voice<br><span>Machine</span></h1>
                <p class="hero-subtitle">
                    Make the machine remember a voice. Encode identity, shape time,
                    and render human-grade speech from the local synthesis core.
                </p>
            </div>
            <div class="hero-visual" aria-hidden="true">
                <div class="noir-sun"></div>
                <div class="horizon-line"></div>
                <div class="visual-code">
                    VOCODER // ACTIVE<br>
                    LATENT CHANNEL // STABLE<br>
                    SIGNAL CLASS // HUMAN
                </div>
            </div>
            <div class="hero-meta" aria-label="Studio status">
                <span class="status-pill"><span class="status-dot"></span>CORE ONLINE</span>
                <span class="status-pill">COMPUTE // {DEVICE.upper()}</span>
                <span class="status-pill">MODEL // V3</span>
                <span class="status-pill">NETWORK // LOCAL</span>
            </div>
        </section>
        """
    )

    with gr.Row(elem_classes=["settings-row"]):
        with gr.Column(scale=7, min_width=360):
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML(
                    """
                    <p class="section-eyebrow">SYS.01 / SYNTH CORE</p>
                    <h2 class="section-title">Calibrate the engine</h2>
                    <p class="section-copy">Select the neural array, language matrix, and artifact format.</p>
                    """
                )
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=[MODEL_V3, MODEL_TURBO],
                        value=MODEL_V3,
                        label="Voice model",
                        elem_id="model-choice",
                    )
                    language_id = gr.Dropdown(
                        choices=[
                            (name, code)
                            for code, name in sorted(
                                SUPPORTED_LANGUAGES.items(),
                                key=lambda item: item[1],
                            )
                        ],
                        value="en",
                        label="Language",
                        elem_id="language-choice",
                    )
                    output_format = gr.Dropdown(
                        choices=[FORMAT_MP3, FORMAT_WAV],
                        value=FORMAT_MP3,
                        label="Output format",
                        elem_id="format-choice",
                    )
                model_status = gr.Markdown(
                    "**V3 selected:** best naturalness, voice similarity, and stability. "
                    "Turbo-only bracketed tags are hidden and removed before synthesis.",
                    elem_classes=["model-status"],
                )

        with gr.Column(scale=5, min_width=320):
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML(
                    """
                    <p class="section-eyebrow">SYS.02 / IDENTITY CAPTURE</p>
                    <h2 class="section-title">Encode a voice</h2>
                    <p class="section-copy">Feed the machine a clean reference signal to map speaker identity.</p>
                    """
                )
                gr.HTML(
                    """
                    <div class="voiceprint-shell">
                        <canvas id="voiceprint-canvas" aria-label="Animated voiceprint signal"></canvas>
                        <div class="voiceprint-label">BIOMETRIC VOICEPRINT // LIVE</div>
                        <div class="voiceprint-readout" id="voiceprint-status">AWAITING IDENTITY SIGNAL</div>
                    </div>
                    """
                )
                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Voice reference",
                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
                    elem_id="reference-audio",
                )
                gr.HTML(
                    """
                    <div class="voice-tip">
                    <strong>OPTIMAL SIGNAL:</strong> 8–20 seconds / one speaker /
                    minimal noise / no music.
                    </div>
                    """
                )

    with gr.Tabs(elem_classes=["tabs-shell"]):
        with gr.Tab("01 / SYNTHESIZE"):
            with gr.Row():
                with gr.Column(scale=7, min_width=380):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">SYS.03 / TRANSMISSION</p>
                            <h2 class="section-title">Write the signal</h2>
                            <p class="section-copy">Compose the transmission, fracture time, then initiate synthesis.</p>
                            """
                        )
                        text = gr.Textbox(
                            value=(
                                "Welcome to Chatterbox Studio. [pause 1s] "
                                "Create something remarkable."
                            ),
                            label="Script",
                            placeholder="Type or paste your script…",
                            max_lines=18,
                            elem_id="main_textbox",
                        )

                        gr.Markdown(
                            "**Temporal markers** / insert at cursor",
                            elem_classes=["tag-heading"],
                        )
                        with gr.Row(elem_classes=["tag-container"]):
                            for pause_tag in PAUSE_BUTTONS:
                                pause_btn = gr.Button(
                                    pause_tag,
                                    elem_classes=["tag-btn"],
                                )
                                pause_btn.click(
                                    fn=None,
                                    inputs=[pause_btn, text],
                                    outputs=text,
                                    js=INSERT_TAG_JS,
                                )

                        with gr.Group(visible=False) as tag_panel:
                            gr.Markdown(
                                "**Performance modifiers** / Turbo array only",
                                elem_classes=["tag-heading"],
                            )
                            with gr.Row(elem_classes=["tag-container"]):
                                for tag in TURBO_TAGS:
                                    btn = gr.Button(tag, elem_classes=["tag-btn"])
                                    btn.click(
                                        fn=None,
                                        inputs=[btn, text],
                                        outputs=text,
                                        js=INSERT_TAG_JS,
                                    )

                        with gr.Row(elem_classes=["action-row"]):
                            run_btn = gr.Button(
                                "Initiate synthesis  //",
                                variant="primary",
                                elem_id="generate-btn",
                            )

                with gr.Column(scale=5, min_width=330):
                    with gr.Group(elem_classes=["studio-card", "output-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">SYS.04 / AUDIO ARTIFACT</p>
                            <h2 class="section-title">Hear the machine</h2>
                            <p class="section-copy">Audit the generated artifact. Every render is archived locally.</p>
                            """
                        )
                        gr.HTML(
                            """
                            <div id="synthesis-reactor">
                                <canvas id="reactor-canvas" aria-label="Synthesis reactor telemetry"></canvas>
                                <div class="reactor-state" id="reactor-status">CORE STANDBY</div>
                                <div class="reactor-core"><span id="reactor-percent">000</span></div>
                                <div class="reactor-log">
                                    <span id="reactor-log-line">AWAITING TRANSMISSION</span>
                                    <span>CBX/49</span>
                                </div>
                            </div>
                            <div class="pipeline-stages" aria-label="Synthesis pipeline">
                                <div class="pipeline-stage">TEXT PARSED</div>
                                <div class="pipeline-stage">VOICE ENCODED</div>
                                <div class="pipeline-stage">SYNTHESIS</div>
                                <div class="pipeline-stage">MASTERING</div>
                                <div class="pipeline-stage">ARCHIVE</div>
                            </div>
                            """
                        )
                        audio_output = gr.Audio(
                            label="Generated audio",
                            elem_id="audio-output",
                            buttons=["download"],
                        )
                        cover_output = gr.Image(
                            label="Generated artifact cover",
                            visible=False,
                            interactive=False,
                            elem_classes=["cover-art"],
                        )
                        gr.Markdown(
                            f"**Saved automatically**  \n`{OUTPUT_DIR}`",
                            elem_classes=["output-note"],
                        )

                        with gr.Accordion(
                            "Advanced synthesis parameters",
                            open=False,
                            elem_classes=["advanced-card"],
                        ):
                            seed_num = gr.Number(
                                value=0,
                                label="Random seed (0 for random)",
                            )
                            temp = gr.Slider(
                                0.05,
                                2.0,
                                step=.05,
                                label="Temperature",
                                value=0.8,
                            )
                            exaggeration = gr.Slider(
                                0.25,
                                2.0,
                                step=0.05,
                                label="V3 expressiveness (neutral = 0.5)",
                                value=0.5,
                            )
                            cfg_weight = gr.Slider(
                                0.0,
                                1.0,
                                step=0.05,
                                label="V3 guidance / pacing",
                                value=0.5,
                            )
                            top_p = gr.Slider(
                                0.00,
                                1.00,
                                step=0.01,
                                label="Top P",
                                value=1.0,
                            )
                            top_k = gr.Slider(
                                0,
                                1000,
                                step=10,
                                label="Turbo Top K",
                                value=1000,
                            )
                            repetition_penalty = gr.Slider(
                                1.00,
                                2.00,
                                step=0.05,
                                label="Repetition penalty",
                                value=1.2,
                            )
                            min_p = gr.Slider(
                                0.00,
                                1.00,
                                step=0.01,
                                label="Min P (0 disables)",
                                value=0.05,
                            )
                            norm_loudness = gr.Checkbox(
                                value=True,
                                label="Turbo loudness normalization (-27 LUFS)",
                            )

        with gr.Tab("02 / BATCH ARRAY"):
            with gr.Row():
                with gr.Column(scale=7, min_width=380):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <div class="queue-icon" aria-hidden="true">//</div>
                            <p class="section-eyebrow">SYS.05 / BATCH ARRAY</p>
                            <h2 class="section-title">Queue the transmissions</h2>
                            <p class="section-copy">Load multiple text artifacts. The array will synthesize them in sequence.</p>
                            """
                        )
                        txt_files = gr.File(
                            label="Text file queue",
                            file_types=[".txt"],
                            file_count="multiple",
                            type="filepath",
                            elem_id="batch-file-input",
                        )
                        text_status = gr.HTML(
                            '<div class="array-standby">ARRAY EMPTY // LOAD .TXT ARTIFACTS</div>',
                            elem_classes=["queue-copy"],
                        )
                        with gr.Row(elem_classes=["action-row"]):
                            queue_btn = gr.Button(
                                "Execute batch array  //",
                                variant="primary",
                                elem_id="queue-btn",
                            )

                with gr.Column(scale=5, min_width=330):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">SYS.06 / ARRAY TELEMETRY</p>
                            <h2 class="section-title">Monitor the artifacts</h2>
                            <p class="section-copy">Completed transmissions remain secured on this machine.</p>
                            """
                        )
                        queue_status = gr.HTML(
                            '<div class="array-standby">TELEMETRY STANDBY // NO ACTIVE BATCH</div>',
                            elem_classes=["output-note"],
                        )

            txt_files.change(
                fn=load_text_files,
                inputs=txt_files,
                outputs=[text, text_status],
            )

    gr.HTML(
        f"""
        <div id="command-strip" aria-label="Persistent synthesis command status">
            <span class="command-brand">CBX/49</span>
            <span>CORE <strong class="command-status" id="cmd-status">STANDBY</strong></span>
            <span class="command-strip-divider"></span>
            <span>MODEL <strong id="cmd-model">V3</strong></span>
            <span>LANG <strong id="cmd-language">EN</strong></span>
            <span>FORMAT <strong id="cmd-format">MP3 / 160K</strong></span>
            <span class="command-strip-divider"></span>
            <span class="command-hide-mobile">COMPUTE <strong>{DEVICE.upper()}</strong></span>
            <span class="command-hide-mobile">ARCHIVE <strong>{OUTPUT_DIR.name.upper()}</strong></span>
        </div>
        """
    )

    model_choice.change(
        fn=model_ui_state,
        inputs=model_choice,
        outputs=[model_status, tag_panel],
        show_progress=False,
    )

    run_btn.click(
        fn=generate,
        inputs=[
            model_choice,
            language_id,
            text,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
            exaggeration,
            cfg_weight,
            output_format,
        ],
        outputs=[audio_output, cover_output],
    )

    queue_btn.click(
        fn=process_file_queue,
        inputs=[
            txt_files,
            model_choice,
            output_format,
            language_id,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
            exaggeration,
            cfg_weight,
        ],
        outputs=queue_status,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        share=False,
        css=CUSTOM_CSS + NEON_NOIR_CSS + DECK_CSS,
        js=APP_JS,
        head=GLOBAL_HEAD_STYLE,
        favicon_path=FAVICON_PATH,
    )
