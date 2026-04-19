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

---

## Web UI

A browser-based UI is available as an alternative to the CLI and tkinter GUI. It runs a local web server accessible from any machine on your network.

```bash
python3 web_server.py
```

Then open **http://localhost:6800** in your browser.

Features:
- Drag-and-drop video upload
- Options panel (Whisper model, top N, keywords, LLM toggle)
- Live progress log streamed in real time
- Results panel with download links and inline "why chosen" reports
- Job queue — submit multiple videos and track each job's status

---

## Server Deployment (Docker)

Run the web UI headlessly on a Linux server using Docker Compose.

### Prerequisites

- Docker and Docker Compose installed on the server
- Port 6800 open in your firewall

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/jxv133-sys/local-clipper.git
cd local-clipper

# 2. Start the services
docker compose up -d

# 3. Open the web UI
# From the server:  http://localhost:6800
# From another machine: http://<server-ip>:6800
```

The `app` service builds from the local `Dockerfile` and mounts `./uploads` and `./output` as volumes so files persist across restarts.

To stop:

```bash
docker compose down
```

To rebuild after code changes:

```bash
docker compose up -d --build
```

---

## Ollama Setup (LLM Scoring)

Ollama provides local LLM scoring. It's optional — the pipeline works without it.

### Install Ollama on Ubuntu

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull a model

```bash
# Fast and lightweight — good default choice
ollama pull llama3.2:1b

# Better instruction following, still runs on CPU
ollama pull llama3.2:3b

# Best quality on CPU (needs ~5GB RAM)
ollama pull llama3.1:8b

# Fastest possible — use if speed matters more than accuracy
ollama pull qwen2.5:0.5b

# Good balance of speed and quality
ollama pull qwen2.5:3b

# Excellent instruction following, very efficient
ollama pull phi3.5
```

### Recommended models for this use case

The pipeline asks each model to rate a short text segment 1–10 for clip-worthiness. You want a model that follows simple instructions reliably and responds quickly — deep reasoning isn't needed.

| Model | RAM | Speed | Quality | Best for |
|-------|-----|-------|---------|----------|
| `qwen2.5:0.5b` | ~1GB | Very fast | Decent | Low-RAM servers, quick jobs |
| `llama3.2:1b` | ~2GB | Fast | Good | **Default recommendation** |
| `qwen2.5:3b` | ~2GB | Fast | Good | Alternative to llama3.2:1b |
| `llama3.2:3b` | ~3GB | Moderate | Better | Better accuracy, still lightweight |
| `phi3.5` | ~3GB | Moderate | Very good | Excellent instruction following |
| `llama3.1:8b` | ~5GB | Slower | Best | Highest quality scoring |

**Recommendation**: start with `llama3.2:1b` for speed, upgrade to `phi3.5` or `llama3.1:8b` if you want more accurate scoring and have the RAM.

To use a specific model:

```bash
python3 main.py video.mp4 --llm --llm-model llama3.2:1b
# or via the web UI: enable LLM scoring and enter the model name in the options panel
```

### Run Ollama as a systemd service (starts on boot)

Ollama's install script sets this up automatically. To verify:

```bash
systemctl status ollama
```

To start/stop manually:

```bash
sudo systemctl start ollama
sudo systemctl stop ollama
```

### Connect the pipeline to a remote Ollama instance

If Ollama is running on a different machine, pass its address via `--llm-endpoint`:

```bash
python3 main.py video.mp4 --llm --llm-endpoint http://192.168.1.100:11434/api/generate
```

When using Docker Compose, the `app` service connects to the `ollama` service automatically via the `OLLAMA_HOST` environment variable. No extra configuration is needed inside the app.

If you bind Ollama to a different host port, set `OLLAMA_HOST` before running the CLI or web server, for example:

```bash
export OLLAMA_HOST=http://localhost:11435
python3 main.py video.mp4 --llm
```
**Note:** Make sure to pull your desired LLM model in the Ollama container first:

```bash
docker compose exec ollama ollama pull llama3
```
---

## Accessing from another machine

The web server binds to `0.0.0.0:6800` by default, so it's reachable from any device on the same network:

```
http://<server-ip>:6800
```

To find your server's IP:

```bash
# Linux
ip addr show | grep "inet " | grep -v 127.0.0.1

# macOS
ipconfig getifaddr en0
```

If you're behind a firewall, open port 6800:

```bash
# Ubuntu (ufw)
sudo ufw allow 6800/tcp
```

---

## Troubleshooting

**FFmpeg not found**

```
AudioExtractionError: FFmpeg is not installed or not accessible on PATH
```

Install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Docker: already included in the image

---

**Whisper model download fails (SSL error on macOS)**

The pipeline includes an automatic SSL fix using `certifi`. If you still see SSL errors:

```bash
pip3 install certifi
/Applications/Python\ 3.x/Install\ Certificates.command
```

---

**No clips generated**

- The video may have no speech — try a video with clear dialogue
- Lower `--top-n` if the video is short
- Check that FFmpeg can read the video format: `ffprobe input.mp4`

---

**Ollama connection refused**

```
LLMScoringError: LLM endpoint unreachable at 'http://localhost:11434/api/generate'
```

- Make sure Ollama is running: `ollama serve` or `systemctl start ollama`
- Make sure you've pulled a model: `ollama pull llama3`
- LLM scoring is optional — omit `--llm` to skip it

---

**LLM returns empty responses**

```
LLM returned no parseable SCORE for window at 123.4s; defaulting to 0.0. Response: ''
```

- The model may not be loaded: `ollama pull llama3` (or your chosen model)
- Try a smaller model if the current one is too slow: `--llm-model llama3.2:1b`
- Check Ollama logs: `ollama serve` in a separate terminal to see error messages
- LLM scoring will fall back to text+audio scoring only

---

**Docker: port 6800 already in use**

Change the host port in `docker-compose.yml`:

```yaml
ports:
  - "7000:6800"   # maps host port 7000 to container port 6800
```

Then access the UI at `http://localhost:7000`.
