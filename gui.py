"""
Video Highlight Generator — GUI
Run with: python3 gui.py
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, font, messagebox, scrolledtext, ttk

# Ensure FFmpeg (Homebrew) is on PATH regardless of how the script is launched
_HOMEBREW_BIN = "/opt/homebrew/bin"
if _HOMEBREW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _HOMEBREW_BIN + ":" + os.environ.get("PATH", "")

# Fix SSL certificate verification on macOS Python.org builds
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

# ---------------------------------------------------------------------------
# Pipeline imports (same as main.py)
# ---------------------------------------------------------------------------
from config import Config
from pipeline.audio_extractor import extract_audio
from pipeline.clip_extractor import extract_clips
from pipeline.clip_selector import select_clips
from pipeline.exceptions import PipelineError
from pipeline.report_generator import generate_report
from pipeline.scorer import score_segments
from pipeline.subtitle_generator import generate_subtitles
from pipeline.transcriber import transcribe


# ---------------------------------------------------------------------------
# Colours / style
# ---------------------------------------------------------------------------
BG = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT = "#7c6af7"
ACCENT2 = "#5a9cf8"
TEXT = "#cdd6f4"
TEXT_DIM = "#6c7086"
SUCCESS = "#a6e3a1"
ERROR = "#f38ba8"
WARNING = "#fab387"
BORDER = "#45475a"

STAGES = [
    "AudioExtractor",
    "Transcriber",
    "Scorer",
    "ClipSelector",
    "ClipExtractor",
    "SubtitleGenerator",
    "ReportGenerator",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 86400.0


# ---------------------------------------------------------------------------
# Pipeline runner (runs in a background thread)
# ---------------------------------------------------------------------------

def run_pipeline_thread(video_path: str, config: Config, msg_queue: queue.Queue) -> None:
    """Run the full pipeline and post progress messages to msg_queue."""

    def log(text: str, level: str = "info") -> None:
        msg_queue.put(("log", level, text))

    def stage_start(name: str) -> None:
        msg_queue.put(("stage_start", name))

    def stage_done(name: str, elapsed: float) -> None:
        msg_queue.put(("stage_done", name, elapsed))

    import time

    try:
        # Stage 1
        stage_start("AudioExtractor")
        t0 = time.time()
        wav_path = extract_audio(config, video_path)
        stage_done("AudioExtractor", time.time() - t0)

        # Stage 2
        stage_start("Transcriber")
        t0 = time.time()
        transcript = transcribe(config, wav_path)
        stage_done("Transcriber", time.time() - t0)

        if not transcript.segments:
            log("⚠ No speech detected — scoring on audio energy only", "warning")

        # Stage 3
        stage_start("Scorer")
        t0 = time.time()
        scored_segments = score_segments(config, transcript, wav_path)
        stage_done("Scorer", time.time() - t0)

        if not scored_segments:
            raise PipelineError("No segments to score. The video may have no audio content.")

        # Stage 4
        stage_start("ClipSelector")
        t0 = time.time()
        video_duration = _get_video_duration(video_path)
        clips = select_clips(config, scored_segments, transcript, video_duration)
        stage_done("ClipSelector", time.time() - t0)

        if not clips:
            raise PipelineError("No clips selected. Try lowering Top N or check the video.")

        log(f"  Selected {len(clips)} clip(s)")

        # Stage 5
        stage_start("ClipExtractor")
        t0 = time.time()
        clip_paths = extract_clips(config, clips, video_path)
        stage_done("ClipExtractor", time.time() - t0)

        # Stage 6
        stage_start("SubtitleGenerator")
        t0 = time.time()
        final_paths = generate_subtitles(config, clips, transcript, clip_paths)
        stage_done("SubtitleGenerator", time.time() - t0)

        # Stage 7
        stage_start("ReportGenerator")
        t0 = time.time()
        report_paths: list[str] = []
        for clip, clip_path in zip(clips, final_paths):
            rp = generate_report(clip, scored_segments, transcript, clip_path, config)
            report_paths.append(rp)
        stage_done("ReportGenerator", time.time() - t0)

        # Clean up temp dir
        shutil.rmtree(config.work_dir, ignore_errors=True)

        msg_queue.put(("done", final_paths, report_paths))

    except PipelineError as exc:
        msg_queue.put(("error", str(exc)))
    except Exception as exc:
        msg_queue.put(("error", f"Unexpected error: {exc}"))


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class HighlightGeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Highlight Generator")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(720, 600)

        self._msg_queue: queue.Queue = queue.Queue()
        self._running = False
        self._stage_index = 0

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Title bar
        title_frame = tk.Frame(self, bg=ACCENT, pady=10)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="🎬  Video Highlight Generator",
                 bg=ACCENT, fg="white",
                 font=("Helvetica", 16, "bold")).pack()

        # Main content
        content = tk.Frame(self, bg=BG, padx=20, pady=16)
        content.pack(fill="both", expand=True)

        # Left column: file + options
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="y", padx=(0, 16))

        # Right column: log + results
        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_file_section(left)
        self._build_options_section(left)
        self._build_run_section(left)
        self._build_log_section(right)
        self._build_results_section(right)

    def _section_label(self, parent, text: str) -> None:
        tk.Label(parent, text=text, bg=BG, fg=ACCENT2,
                 font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(12, 4))

    def _build_file_section(self, parent) -> None:
        self._section_label(parent, "INPUT VIDEO")

        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x")

        self._file_var = tk.StringVar(value="No file selected")
        file_entry = tk.Entry(row, textvariable=self._file_var,
                              bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                              relief="flat", font=("Helvetica", 10), width=32)
        file_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        tk.Button(row, text="Browse", command=self._browse_file,
                  bg=ACCENT, fg="white", relief="flat",
                  font=("Helvetica", 10, "bold"),
                  padx=12, pady=4, cursor="hand2").pack(side="left")

    def _build_options_section(self, parent) -> None:
        self._section_label(parent, "OPTIONS")

        frame = tk.Frame(parent, bg=SURFACE, padx=14, pady=12, relief="flat")
        frame.pack(fill="x")

        def row(label: str, widget_fn):
            r = tk.Frame(frame, bg=SURFACE)
            r.pack(fill="x", pady=4)
            tk.Label(r, text=label, bg=SURFACE, fg=TEXT,
                     font=("Helvetica", 10), width=16, anchor="w").pack(side="left")
            widget_fn(r)

        # Whisper model
        self._whisper_var = tk.StringVar(value="base")
        row("Whisper model:", lambda r: ttk.Combobox(
            r, textvariable=self._whisper_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly", width=10
        ).pack(side="left"))

        # Top N clips
        self._topn_var = tk.IntVar(value=3)
        row("Top N clips:", lambda r: tk.Spinbox(
            r, from_=1, to=20, textvariable=self._topn_var,
            bg=SURFACE, fg=TEXT, buttonbackground=SURFACE,
            relief="flat", width=5, font=("Helvetica", 10)
        ).pack(side="left"))

        # Output dir
        self._outdir_var = tk.StringVar(value="")
        def outdir_row(r):
            e = tk.Entry(r, textvariable=self._outdir_var,
                         bg=BG, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Helvetica", 10), width=18)
            e.pack(side="left", ipady=4, padx=(0, 6))
            tk.Button(r, text="…", command=self._browse_outdir,
                      bg=BORDER, fg=TEXT, relief="flat",
                      font=("Helvetica", 10), padx=6, cursor="hand2").pack(side="left")
        row("Output dir:", outdir_row)

        # Keywords
        self._keywords_var = tk.StringVar(value="crazy, important, watch this, incredible")
        row("Keywords:", lambda r: tk.Entry(
            r, textvariable=self._keywords_var,
            bg=BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Helvetica", 10), width=26
        ).pack(side="left", ipady=4))

        # LLM toggle
        self._llm_var = tk.BooleanVar(value=False)
        def llm_row(r):
            tk.Checkbutton(r, variable=self._llm_var, text="Enable (Ollama)",
                           bg=SURFACE, fg=TEXT, selectcolor=SURFACE,
                           activebackground=SURFACE, activeforeground=TEXT,
                           font=("Helvetica", 10)).pack(side="left")
        row("LLM scoring:", llm_row)

        # LLM model
        self._llm_model_var = tk.StringVar(value="llama3")
        row("LLM model:", lambda r: tk.Entry(
            r, textvariable=self._llm_model_var,
            bg=BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", font=("Helvetica", 10), width=14
        ).pack(side="left", ipady=4))

    def _build_run_section(self, parent) -> None:
        frame = tk.Frame(parent, bg=BG, pady=12)
        frame.pack(fill="x")

        self._run_btn = tk.Button(
            frame, text="▶  Generate Highlights",
            command=self._start_pipeline,
            bg=ACCENT, fg="white", relief="flat",
            font=("Helvetica", 12, "bold"),
            padx=16, pady=8, cursor="hand2",
        )
        self._run_btn.pack(fill="x")

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Highlight.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            frame, variable=self._progress_var,
            maximum=len(STAGES), style="Highlight.Horizontal.TProgressbar",
            length=280,
        )
        self._progress_bar.pack(fill="x", pady=(8, 0))

        self._stage_label = tk.Label(frame, text="", bg=BG, fg=TEXT_DIM,
                                     font=("Helvetica", 9))
        self._stage_label.pack(anchor="w")

    def _build_log_section(self, parent) -> None:
        self._section_label(parent, "PROGRESS LOG")

        self._log = scrolledtext.ScrolledText(
            parent, bg=SURFACE, fg=TEXT,
            font=("Courier", 10), relief="flat",
            state="disabled", height=18, wrap="word",
        )
        self._log.pack(fill="both", expand=True)

        # Tag colours
        self._log.tag_config("info",    foreground=TEXT)
        self._log.tag_config("success", foreground=SUCCESS)
        self._log.tag_config("warning", foreground=WARNING)
        self._log.tag_config("error",   foreground=ERROR)
        self._log.tag_config("stage",   foreground=ACCENT2, font=("Courier", 10, "bold"))
        self._log.tag_config("dim",     foreground=TEXT_DIM)

    def _build_results_section(self, parent) -> None:
        self._section_label(parent, "OUTPUT CLIPS")

        self._results_frame = tk.Frame(parent, bg=SURFACE, padx=10, pady=8)
        self._results_frame.pack(fill="x")

        self._results_placeholder = tk.Label(
            self._results_frame,
            text="Clips will appear here after processing.",
            bg=SURFACE, fg=TEXT_DIM, font=("Helvetica", 10),
        )
        self._results_placeholder.pack(anchor="w")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if path:
            self._file_var.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self._outdir_var.set(path)

    def _start_pipeline(self) -> None:
        if self._running:
            return

        video_path = self._file_var.get()
        if not video_path or video_path == "No file selected" or not os.path.exists(video_path):
            messagebox.showerror("No video", "Please select a valid video file first.")
            return

        # Build config
        work_dir = tempfile.mkdtemp(prefix="highlight_")

        output_dir = self._outdir_var.get().strip()
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "highlights")

        cfg = Config(work_dir=work_dir)
        cfg.output_dir = output_dir
        cfg.whisper_model = self._whisper_var.get()
        cfg.top_n_clips = self._topn_var.get()
        cfg.llm_enabled = self._llm_var.get()
        cfg.llm_model = self._llm_model_var.get()

        raw_kw = self._keywords_var.get()
        if raw_kw.strip():
            cfg.keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]

        # Reset UI
        self._clear_log()
        self._clear_results()
        self._progress_var.set(0)
        self._stage_index = 0
        self._stage_label.config(text="")
        self._run_btn.config(state="disabled", text="⏳  Running…")
        self._running = True

        self._log_write(f"Input:   {video_path}\n", "dim")
        self._log_write(f"Output:  {output_dir}\n", "dim")
        self._log_write(f"Whisper: {cfg.whisper_model}  |  Top N: {cfg.top_n_clips}  |  LLM: {'on' if cfg.llm_enabled else 'off'}\n\n", "dim")

        # Launch background thread
        t = threading.Thread(
            target=run_pipeline_thread,
            args=(video_path, cfg, self._msg_queue),
            daemon=True,
        )
        t.start()

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_message(self, msg: tuple) -> None:
        kind = msg[0]

        if kind == "log":
            _, level, text = msg
            self._log_write(text + "\n", level)

        elif kind == "stage_start":
            _, name = msg
            self._log_write(f"[{name}] Starting…\n", "stage")
            self._stage_label.config(text=f"Stage: {name}")

        elif kind == "stage_done":
            _, name, elapsed = msg
            self._log_write(f"[{name}] ✓ Done in {elapsed:.1f}s\n", "success")
            self._stage_index += 1
            self._progress_var.set(self._stage_index)

        elif kind == "done":
            _, final_paths, report_paths = msg
            self._log_write("\n✓ Pipeline complete!\n", "success")
            self._running = False
            self._run_btn.config(state="normal", text="▶  Generate Highlights")
            self._stage_label.config(text="Complete ✓")
            self._show_results(final_paths, report_paths)

        elif kind == "error":
            _, err = msg
            self._log_write(f"\n✗ Error: {err}\n", "error")
            self._running = False
            self._run_btn.config(state="normal", text="▶  Generate Highlights")
            self._stage_label.config(text="Failed ✗")

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log_write(self, text: str, tag: str = "info") -> None:
        self._log.config(state="normal")
        self._log.insert("end", text, tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self) -> None:
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        for widget in self._results_frame.winfo_children():
            widget.destroy()
        self._results_placeholder = tk.Label(
            self._results_frame,
            text="Clips will appear here after processing.",
            bg=SURFACE, fg=TEXT_DIM, font=("Helvetica", 10),
        )
        self._results_placeholder.pack(anchor="w")

    def _show_results(self, final_paths: list[str], report_paths: list[str]) -> None:
        for widget in self._results_frame.winfo_children():
            widget.destroy()

        for i, (clip_path, report_path) in enumerate(zip(final_paths, report_paths), 1):
            row = tk.Frame(self._results_frame, bg=SURFACE, pady=4)
            row.pack(fill="x")

            tk.Label(row, text=f"#{i}", bg=SURFACE, fg=ACCENT,
                     font=("Helvetica", 10, "bold"), width=3).pack(side="left")

            name = os.path.basename(clip_path)
            tk.Label(row, text=name, bg=SURFACE, fg=TEXT,
                     font=("Helvetica", 10)).pack(side="left", padx=(4, 8))

            tk.Button(row, text="📂 Open",
                      command=lambda p=clip_path: self._open_file(p),
                      bg=ACCENT2, fg="white", relief="flat",
                      font=("Helvetica", 9), padx=8, pady=2,
                      cursor="hand2").pack(side="left", padx=(0, 4))

            tk.Button(row, text="📄 Why chosen",
                      command=lambda p=report_path: self._open_file(p),
                      bg=BORDER, fg=TEXT, relief="flat",
                      font=("Helvetica", 9), padx=8, pady=2,
                      cursor="hand2").pack(side="left")

        # Open output folder button
        if final_paths:
            out_dir = os.path.dirname(final_paths[0])
            tk.Button(
                self._results_frame,
                text=f"📁  Open output folder",
                command=lambda: self._open_folder(out_dir),
                bg=ACCENT, fg="white", relief="flat",
                font=("Helvetica", 10, "bold"),
                padx=12, pady=6, cursor="hand2",
            ).pack(anchor="w", pady=(10, 0))

    def _open_file(self, path: str) -> None:
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path])

    def _open_folder(self, path: str) -> None:
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform == "win32":
            subprocess.run(["explorer", path])
        else:
            subprocess.run(["xdg-open", path])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = HighlightGeneratorApp()
    app.mainloop()
