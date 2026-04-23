# CLI Cheat Sheet — Video Highlight Generator

---

## `main.py` — Command-line pipeline

```
python3 main.py [input_video] [options]
python3 main.py --url <youtube_url> [options]
```

### Input

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `input_video` | positional | — | Path to a local video file. Omit when using `--url`. |
| `--url URL` | string | — | YouTube URL to download and clip automatically. |
| `--quality N` | int | `720` | Max resolution for YouTube downloads. Choices: `360`, `480`, `720`, `1080`. Lower = faster download. |

### Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir DIR` | string | `<video_folder>/highlights` | Directory where clips and reports are saved. |
| `--no-subtitles` | flag | off | Skip burning subtitles into clips. SRT files are still written alongside each clip. |

### Transcription

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--whisper-model MODEL` | string | `base` | Whisper model size. Choices: `tiny`, `base`, `small`, `medium`, `large`. Larger = more accurate but slower. |

### Clip selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--top-n N` | int | `5` | Number of highlight clips to generate. |
| `--keywords KW [KW ...]` | list | *(built-in list)* | Space-separated keywords that boost segment scores. Replaces the default keyword list entirely. |

### LLM scoring (Ollama)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--llm` | flag | off | Enable LLM scoring via Ollama. Adds hook detection and window scoring. |
| `--llm-model MODEL` | string | `llama3` | Ollama model name to use (must be pulled first). |
| `--llm-endpoint URL` | string | `http://localhost:11434/api/generate` | Ollama API endpoint. Override with `OLLAMA_HOST` env var or this flag. |

---

### Quick examples

```bash
# Basic — local file, 5 clips, no LLM
python3 main.py stream.mp4

# More clips, better transcription
python3 main.py stream.mp4 --top-n 8 --whisper-model small

# YouTube download at 480p, 3 clips, skip subtitles (faster)
python3 main.py --url https://youtu.be/gzL2xoZLg9I --quality 480 --top-n 3 --no-subtitles

# Full LLM pipeline with custom output dir
python3 main.py stream.mp4 --llm --llm-model llama3.2:3b --top-n 6 --output-dir ~/clips

# Custom keywords
python3 main.py stream.mp4 --keywords clutch insane "no way" "let's go"

# YouTube + LLM
python3 main.py --url https://youtu.be/gzL2xoZLg9I --llm --llm-model llama3.2:1b --no-subtitles
```

---

## `web_server.py` — Web UI server

```
python3 web_server.py [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host HOST` | string | `0.0.0.0` | Host to bind to. Use `127.0.0.1` to restrict to localhost only. |
| `--port PORT` | int | `6800` | Port to listen on. Access at `http://localhost:6800`. |
| `--uploads-dir DIR` | string | `./uploads` | Directory where uploaded videos are temporarily stored. |
| `--output-dir DIR` | string | `./output` | Default directory for completed clips. |
| `--no-cleanup-uploads` | flag | off | Keep uploaded source videos after job completes. By default they are deleted to save disk space. |

### Quick examples

```bash
# Default — listen on all interfaces, port 6800
python3 web_server.py

# Custom port
python3 web_server.py --port 8080

# Localhost only (more secure)
python3 web_server.py --host 127.0.0.1

# Keep uploaded videos (useful for debugging)
python3 web_server.py --no-cleanup-uploads

# Custom storage paths
python3 web_server.py --uploads-dir /data/uploads --output-dir /data/clips
```

---

## Docker Compose

```bash
# Start everything
docker compose up -d

# Pull an LLM model into the running Ollama container
docker compose exec ollama ollama pull llama3.2:1b   # ~1.3 GB, fast
docker compose exec ollama ollama pull llama3.2:3b   # ~2.0 GB, better quality
docker compose exec ollama ollama pull llama3        # ~4.7 GB, best quality

# Check what models are available
docker compose exec ollama ollama list

# View logs
docker compose logs -f app
docker compose logs -f ollama

# Stop
docker compose down
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOST` | Override the Ollama endpoint. e.g. `http://ollama:11434`. Takes precedence over `--llm-endpoint`. |
| `OMP_NUM_THREADS` | CPU thread count for numpy/scipy operations. Set in docker-compose.yml. |
| `NUMBA_NUM_THREADS` | CPU thread count for numba (used by librosa). Set in docker-compose.yml. |

---

## Whisper model comparison

| Model | Size | Speed | Accuracy | Best for |
|-------|------|-------|----------|----------|
| `tiny` | 75 MB | ⚡⚡⚡⚡ | ★★☆☆ | Quick tests |
| `base` | 145 MB | ⚡⚡⚡ | ★★★☆ | Default, good balance |
| `small` | 465 MB | ⚡⚡ | ★★★★ | Better accuracy |
| `medium` | 1.5 GB | ⚡ | ★★★★ | High accuracy |
| `large` | 3 GB | 🐢 | ★★★★★ | Best accuracy |

---

## LLM model comparison (Ollama)

| Model | Size | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| `llama3.2:1b` | 1.3 GB | ⚡⚡⚡ | ★★☆☆ | Minimum viable, fast |
| `llama3.2:3b` | 2.0 GB | ⚡⚡ | ★★★☆ | Good balance |
| `llama3` | 4.7 GB | ⚡ | ★★★★ | Recommended |
| `mistral` | 4.1 GB | ⚡ | ★★★★ | Good alternative |
| `llama3:70b` | 40 GB | 🐢 | ★★★★★ | Needs 32 GB+ RAM |
