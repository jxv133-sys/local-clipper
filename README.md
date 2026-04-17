# Video Highlight Generator

A local, self-hosted pipeline that automatically finds the best moments in a video and exports them as short highlight clips with burned-in subtitles. Runs entirely offline — no cloud services, no paid APIs.

## How it works

1. **Audio extraction** — FFmpeg extracts a mono 16kHz WAV from the input video
2. **Transcription** — faster-whisper (CTranslate2) transcribes speech with timestamps
3. **Scoring** — each segment is scored on audio energy (RMS), keyword detection, and punctuation
4. **Clip selection** — top N segments are expanded to 20–45s clips, overlaps merged
5. **Clip extraction** — FFmpeg extracts clips from the original video (stream-copy, no quality loss)
6. **Subtitles** — subtitles are rendered with Pillow and burned into each clip
7. **Reports** — a `_why_chosen.txt` file is written alongside each clip explaining the selection

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10+
- FFmpeg (via Homebrew)

## Installation

```bash
# 1. Install FFmpeg
brew install ffmpeg

# 2. Clone the repo
git clone https://github.com/jxv133-sys/local-clipper.git
cd local-clipper

# 3. Install Python dependencies
pip3 install -r requirements.txt
```

## Usage

### GUI (recommended)

```bash
python3 gui.py
```

Opens a window where you can:
- Browse for a video file
- Set options (Whisper model, number of clips, keywords, LLM scoring)
- Watch live progress as each stage runs
- Open output clips and "why chosen" reports directly from the results panel

### CLI

```bash
python3 main.py input.mp4
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--top-n` | `5` | Number of highlight clips to generate |
| `--output-dir` | `<input_folder>/highlights` | Where to save output clips |
| `--keywords` | built-in list | Space-separated words that boost segment scores |
| `--llm` | off | Enable local LLM scoring via Ollama |
| `--llm-model` | `llama3` | Ollama model name |

**Examples:**

```bash
# Generate 3 clips using the small Whisper model
python3 main.py video.mp4 --top-n 3 --whisper-model small

# Custom keywords
python3 main.py video.mp4 --keywords "insane" "no way" "watch this"

# With local LLM scoring (requires Ollama running)
python3 main.py video.mp4 --llm --llm-model llama3
```

## Output

All files are saved to `<input_folder>/highlights/` by default:

```
highlights/
├── clip_1_120s.mp4          # Rank 1 highlight clip (with subtitles)
├── clip_1_120s.srt          # Subtitle file
├── clip_1_120s_why_chosen.txt  # Score breakdown and transcript
├── clip_2_340s.mp4
├── clip_2_340s.srt
├── clip_2_340s_why_chosen.txt
└── ...
```

Each `_why_chosen.txt` explains:
- Clip timing and duration
- Overall score with ASCII bar charts for audio energy, text interest, and LLM rating
- Which keywords were detected
- The full transcript of the clip

## Whisper model sizes

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `tiny` | ~10x faster than base | Lower | ~1GB |
| `base` | baseline | Good | ~1GB |
| `small` | ~2x slower | Better | ~2GB |
| `medium` | ~5x slower | High | ~5GB |
| `large` | ~10x slower | Best | ~10GB |

For highlight detection, `base` or `small` is usually the best tradeoff.

## Optional: LLM scoring

If you have [Ollama](https://ollama.ai) running locally, enable LLM scoring to get a 1–10 clip-worthiness rating per segment:

```bash
# Start Ollama with a model
ollama run llama3

# Run with LLM scoring
python3 main.py video.mp4 --llm
```

## Known limitations

- Transcription speed depends on CPU — a 10-minute video takes ~3-5 minutes on Apple Silicon with the `base` model (faster-whisper with int8 quantization)
- Subtitle font uses system fonts (Helvetica/Arial on macOS); falls back to a basic font if none found
- Stream-copy extraction preserves original quality but may have slightly inaccurate cut points on some codecs; the pipeline automatically re-encodes if the duration is off by more than 1 second
