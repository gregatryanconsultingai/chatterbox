import random
import re
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import gradio as gr
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
OUTPUT_DIR = APP_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def load_text_files(file_paths):
    paths = normalize_file_paths(file_paths)
    if not paths:
        return gr.update(), "Add one or more `.txt` files to the queue."

    documents = []
    queue_lines = []
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
        queue_lines.append(
            f"- `{path.name}` — {len(document):,} characters, "
            f"{section_count:,} sections"
        )

    status = (
        f"**{len(paths):,} file{'s' if len(paths) != 1 else ''} queued** — "
        f"{total_characters:,} characters, {total_sections:,} reading sections"
        + (f", {total_pauses:,} explicit pauses." if total_pauses else ".")
        + "\n\n"
        + "\n".join(queue_lines[:20])
    )
    if len(queue_lines) > 20:
        status += f"\n- …and {len(queue_lines) - 20:,} more"
    return documents[0], status


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

        if output_format == FORMAT_MP3:
            if not FFMPEG_EXE:
                raise RuntimeError(
                    "MP3 encoding requires FFmpeg, but it was not found."
                )
            progress(0.99, desc="Encoding maximum-quality MP3")
            subprocess.run(
                [
                    FFMPEG_EXE,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(working_path),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "160k",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            working_path.unlink(missing_ok=True)
    except Exception:
        working_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise

    progress(1, desc="Finished")
    return str(output_path)


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

    for file_index, path in enumerate(paths):
        try:
            document = read_text_file(path)

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
        except Exception as exc:
            failures.append(f"`{path.name}`: {exc}")

    progress(1, desc="Queue finished")
    status = (
        f"**Completed {len(completed):,} of {len(paths):,} files.**\n\n"
        f"Saved permanently to `{OUTPUT_DIR}`."
    )
    if completed:
        status += "\n\n**Completed sources:**\n" + "\n".join(
            f"- `{filename}`"
            for filename in completed[:20]
        )
        if len(completed) > 20:
            status += f"\n- …and {len(completed) - 20:,} more"
    if failures:
        status += "\n\n**Failed:**\n" + "\n".join(
            f"- {failure}"
            for failure in failures
        )
    if not completed:
        raise gr.Error("No queued files could be generated.")
    return status


with gr.Blocks(title="Chatterbox Studio") as demo:
    gr.HTML(
        f"""
        <section class="hero-shell">
            <div class="hero-copy">
                <div class="brand-mark" aria-hidden="true">◉</div>
                <p class="eyebrow">Local voice lab</p>
                <h1>Chatterbox <span>Studio</span></h1>
                <p class="hero-subtitle">
                    Turn scripts into expressive, studio-quality speech with voice
                    cloning, natural pauses, and effortless batch production.
                </p>
            </div>
            <div class="hero-meta" aria-label="Studio status">
                <span class="status-pill"><span class="status-dot"></span>{DEVICE.upper()} ready</span>
                <span class="status-pill">✦ V3 quality</span>
                <span class="status-pill">⌁ Private &amp; local</span>
            </div>
        </section>
        """
    )

    with gr.Row(elem_classes=["settings-row"]):
        with gr.Column(scale=7, min_width=360):
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML(
                    """
                    <p class="section-eyebrow">01 · Sound engine</p>
                    <h2 class="section-title">Shape your performance</h2>
                    <p class="section-copy">Pick the quality, language, and delivery format for this session.</p>
                    """
                )
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=[MODEL_V3, MODEL_TURBO],
                        value=MODEL_V3,
                        label="Voice model",
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
                    )
                    output_format = gr.Dropdown(
                        choices=[FORMAT_MP3, FORMAT_WAV],
                        value=FORMAT_MP3,
                        label="Output format",
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
                    <p class="section-eyebrow">02 · Voice DNA</p>
                    <h2 class="section-title">Choose a voice</h2>
                    <p class="section-copy">Upload or record a clean reference clip to guide the speaker identity.</p>
                    """
                )
                ref_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Voice reference",
                    value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
                )
                gr.HTML(
                    """
                    <div class="voice-tip">
                        <strong>For the cleanest clone:</strong> use 8–20 seconds of one
                        speaker, with little background noise or music.
                    </div>
                    """
                )

    with gr.Tabs(elem_classes=["tabs-shell"]):
        with gr.Tab("✦ Create"):
            with gr.Row():
                with gr.Column(scale=7, min_width=380):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">03 · Script studio</p>
                            <h2 class="section-title">Write something worth hearing</h2>
                            <p class="section-copy">Compose here, add precise pauses, then render a polished take.</p>
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
                            "**Timing shortcuts** · insert at your cursor",
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
                                "**Turbo performance tags** · sounds and styles",
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
                                "Generate voice  ✦",
                                variant="primary",
                                elem_id="generate-btn",
                            )

                with gr.Column(scale=5, min_width=330):
                    with gr.Group(elem_classes=["studio-card", "output-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">04 · Final take</p>
                            <h2 class="section-title">Listen &amp; refine</h2>
                            <p class="section-copy">Your latest render appears here and is also saved automatically.</p>
                            """
                        )
                        audio_output = gr.Audio(label="Generated audio")
                        gr.Markdown(
                            f"**Saved automatically**  \n`{OUTPUT_DIR}`",
                            elem_classes=["output-note"],
                        )

                        with gr.Accordion(
                            "Fine-tune generation",
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

        with gr.Tab("⇉ Batch queue"):
            with gr.Row():
                with gr.Column(scale=7, min_width=380):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <div class="queue-icon" aria-hidden="true">⇉</div>
                            <p class="section-eyebrow">Batch production</p>
                            <h2 class="section-title">Turn a stack of scripts into audio</h2>
                            <p class="section-copy">Drop in multiple text files and Chatterbox will process them in order.</p>
                            """
                        )
                        txt_files = gr.File(
                            label="Text file queue",
                            file_types=[".txt"],
                            file_count="multiple",
                            type="filepath",
                        )
                        text_status = gr.Markdown(
                            "Add one or more `.txt` files. The first file also opens "
                            "in the Script Studio. Use `[pause]` for one second or a "
                            "timed marker such as `[pause 1.5s]`.",
                            elem_classes=["queue-copy"],
                        )
                        with gr.Row(elem_classes=["action-row"]):
                            queue_btn = gr.Button(
                                "Process entire queue  ⇉",
                                variant="primary",
                                elem_id="queue-btn",
                            )

                with gr.Column(scale=5, min_width=330):
                    with gr.Group(elem_classes=["studio-card"]):
                        gr.HTML(
                            """
                            <p class="section-eyebrow">Queue activity</p>
                            <h2 class="section-title">Production status</h2>
                            <p class="section-copy">Completed files remain safely stored on this computer.</p>
                            """
                        )
                        queue_status = gr.Markdown(
                            f"Ready when you are. Outputs will be saved to  \n`{OUTPUT_DIR}`.",
                            elem_classes=["output-note"],
                        )

            txt_files.change(
                fn=load_text_files,
                inputs=txt_files,
                outputs=[text, text_status],
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
        outputs=audio_output,
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
    ).launch(share=False, css=CUSTOM_CSS)
