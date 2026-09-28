"""Drag-and-drop web UI for whisperX. Run: uv run --with gradio --with "huggingface-hub<1.0" python app.py"""
import os
import tempfile
from functools import lru_cache

import gradio as gr
import whisperx
from faster_whisper.tokenizer import _LANGUAGE_CODES
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer

DEVICE = "cpu"  # no CUDA on Mac


@lru_cache(maxsize=2)
def asr(model_name):
    return whisperx.load_model(model_name, DEVICE, compute_type="int8")


@lru_cache(maxsize=4)
def aligner(lang):
    return whisperx.load_align_model(language_code=lang, device=DEVICE)


@lru_cache(maxsize=1)
def diarizer():
    # token from HF_TOKEN env or `hf auth login`; model terms must be accepted on huggingface.co
    return DiarizationPipeline(token=os.environ.get("HF_TOKEN"), device=DEVICE)


def split_by_speaker(segments, lang):
    """whisperX labels a whole segment with its majority speaker; split wherever the word-level speaker changes."""
    sep = "" if lang in ("zh", "ja") else " "
    out = []
    for seg in segments:
        for w in seg.get("words", []):
            spk = w.get("speaker") or (out[-1]["speaker"] if out else seg.get("speaker"))
            if not out or out[-1]["speaker"] != spk or out[-1]["_seg"] is not seg:
                out.append({"speaker": spk, "start": w.get("start", seg["start"]), "end": seg["end"], "words": [], "_seg": seg})
            cur = out[-1]
            cur["words"].append(w)
            cur["end"] = w.get("end", cur["end"])
    for s in out:
        s["text"] = sep.join(w["word"] for w in s["words"])
        del s["_seg"]
    return out


def transcribe(path, mic, model_name, language, speakers):
    path = path or mic
    if not path:
        raise gr.Error("Upload a file or record from the microphone first.")
    audio = whisperx.load_audio(path)
    result = asr(model_name).transcribe(audio, batch_size=8, language=language or None)
    lang = result["language"]
    model_a, meta = aligner(lang)
    result = whisperx.align(result["segments"], model_a, meta, audio, DEVICE)
    if speakers:
        try:
            result = whisperx.assign_word_speakers(diarizer()(audio), result)
            result["segments"] = split_by_speaker(result["segments"], lang)
        except Exception as e:
            raise gr.Error(
                "Speaker detection needs a Hugging Face token: run `uv run hf auth login`, and accept the terms at "
                f"huggingface.co/pyannote/speaker-diarization-community-1. ({type(e).__name__}: {e})"
            )
    result["language"] = lang

    out = tempfile.mkdtemp()
    opts = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
    for fmt in ("srt", "txt", "json"):
        get_writer(fmt, out)(result, path, opts)
    stem = os.path.splitext(os.path.basename(path))[0]
    files = [os.path.join(out, f"{stem}.{f}") for f in ("srt", "txt", "json")]
    return open(files[1]).read(), files


demo = gr.Interface(
    transcribe,
    [
        gr.File(label="Audio / video", type="filepath"),
        gr.Audio(sources=["microphone"], type="filepath", label="…or record from mic"),
        gr.Dropdown(["tiny", "base", "small", "medium", "large-v3", "turbo"], value="small", label="Model"),
        gr.Dropdown(["", *sorted(_LANGUAGE_CODES)], value="", label="Language (blank = auto-detect, Chinese = zh)"),
        gr.Checkbox(label="Detect speakers"),
    ],
    [gr.Textbox(label="Transcript", lines=15), gr.File(label="Downloads (SRT / TXT / JSON)")],
    title="whisperX",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
