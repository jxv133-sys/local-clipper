# CLI Cheat Sheet — Video Highlight Generator

---

## main.py — Command-line pipeline

```
python3 main.py [input_video] [options]
python3 main.py --url <youtube_url> [options]
```

---

### Input

**`input_video`** *(positional)*
Path to a local video file. Omit when using `--url`.

**`--url URL`**
Download a YouTube video and run the full pipeline on it automatically.

**`--quality 360|480|720|1080`** *(default: 720)*
Max resolution for YouTube downloads. Lower = faster download and processing.
`480` is a good balance for long streams.

---

### Output

**`--output-dir DIR`** *(default: `<video_folder>/highlights`)*
Directory where clips, SRT files, and why-chosen reports are saved.

**`--no-subtitles`**
Skip burning subtitles into the video. SRT files are still written alongside
each clip. Use this during testing — subtitle burn-in is the slowest step.

---

### Transcription

**`--whisper-model tiny|base|small|medium|large`** *(default: `base`)*
Whisper model size. Larger models are more accurate but slower to run.

```
tiny   →  75 MB   fastest, lower accuracy
base   → 145 MB   good default
small  → 465 MB   noticeably better accuracy
medium → 1.5 GB   high accuracy
large  →   3 GB   best accuracy, slow on CPU
```

---

### Clip selection

**`--top-n N`** *(default: 5)*
Number of highlight clips to generate.

**`--keywords KW [KW ...]`**
Space-separated keywords that boost segment scores. Replaces the built-in
keyword list entirely. Wrap multi-word phrases in quotes.

```bash
--keywords clutch insane "no way" "let's go" "oh my god"
```

---

### LLM scoring (Ollama)

**`--llm`**
Enable LLM scoring. Adds hook detection and per-window quality scoring.
Requires Ollama to be running with a model pulled.

**`--llm-model MODEL`** *(default: `llama3`)*
Ollama model name. Must be pulled before use (`ollama pull <model>`).

```
llama3.2:1b  →  1.3 GB   fast, minimum viable
llama3.2:3b  →  2.0 GB   good balance
llama3       →  4.7 GB   recommended
mistral      →  4.1 GB   good alternative
```

**`--llm-endpoint URL`** *(default: `http://localhost:11434/api/generate`)*
Ollama API endpoint. Can also be set via the `OLLAMA_HOST` environment variable.

---

### Quick examples

```bash
# Basic — local file, 5 clips
python3 main.py stream.mp4

# More clips, better transcription
python3 main.py stream.mp4 --top-n 8 --whisper-model small

# YouTube at 480p, 3 clips, no subtitles (fast)
python3 main.py --url https://youtu.be/gzL2xoZLg9I --quality 480 --top-n 3 --no-subtitles

# Full LLM pipeline
python3 main.py stream.mp4 --llm --llm-model llama3.2:3b --top-n 6

# Custom keywords + custom output dir
python3 main.py stream.mp4 --keywords clutch insane "no way" --output-dir ~/clips

# YouTube + LLM, skip subtitles for speed
python3 main.py --url https://youtu.be/gzL2xoZLg9I --llm --llm-model llama3.2:1b --no-subtitles
```

---
---

## web_server.py — Web UI server

```
python3 web_server.py [options]
```

Access the UI at `http://localhost:6800` after starting.

---

**`--port PORT`** *(default: 6800)*
Port to listen on.

**`--host HOST`** *(default: 0.0.0.0)*
Host to bind to. Use `127.0.0.1` to restrict to localhost only.

**`--uploads-dir DIR`** *(default: ./uploads)*
Where uploaded videos are temporarily stored during processing.

**`--output-dir DIR`** *(default: ./output)*
Default directory for completed clips.

**`--no-cleanup-uploads`**
Keep uploaded source videos after a job completes. By default they are
deleted to save disk space.

---

### Quick examples

```bash
# Default
python3 web_server.py

# Custom port
python3 web_server.py --port 8080

# Localhost only (more secure for local use)
python3 web_server.py --host 127.0.0.1

# Keep uploads for debugging
python3 web_server.py --no-cleanup-uploads

# Custom storage paths
python3 web_server.py --uploads-dir /data/uploads --output-dir /data/clips
```

---
---

## Docker Compose

```bash
# Start everything
docker compose up -d

# Pull an LLM model into the running Ollama container
docker compose exec ollama ollama pull llama3.2:1b    # 1.3 GB, fast
docker compose exec ollama ollama pull llama3.2:3b    # 2.0 GB, better
docker compose exec ollama ollama pull llama3         # 4.7 GB, best

# List pulled models
docker compose exec ollama ollama list

# View live logs
docker compose logs -f app
docker compose logs -f ollama

# Stop
docker compose down
```

---
---

## Environment variables

**`OLLAMA_HOST`**
Override the Ollama endpoint URL. Takes precedence over `--llm-endpoint`.
Example: `export OLLAMA_HOST=http://ollama:11434`

**`OMP_NUM_THREADS`**
CPU thread count for numpy/scipy. Set in `docker-compose.yml`.

**`NUMBA_NUM_THREADS`**
CPU thread count for librosa/numba. Set in `docker-compose.yml`.
