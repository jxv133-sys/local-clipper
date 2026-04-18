from dataclasses import dataclass, field


@dataclass
class Config:
    # Paths
    work_dir: str                    # Temporary working directory (set at runtime)
    output_dir: str = "output"       # Where final clips are saved

    # Whisper
    whisper_model: str = "base"      # Whisper model size: tiny/base/small/medium/large

    # Scoring weights (must sum to 1.0 when LLM is enabled)
    # Default: balanced text/audio. When LLM enabled, weights auto-adjust in main.py.
    text_weight: float = 0.5         # Raised from 0.4 — text interest matters more
    audio_weight: float = 0.5        # Lowered from 0.6 — audio alone isn't enough
    llm_weight: float = 0.0          # 0.0 = LLM disabled

    # LLM (optional)
    llm_enabled: bool = False
    llm_endpoint: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3"
    # Maximum number of candidate windows sent to the LLM.
    # The scorer picks the top-scoring moments (spaced >= min_clip_duration apart)
    # and calls the LLM once per window — never once per Whisper segment.
    # Rule of thumb: top_n_clips * 3 gives the LLM enough candidates to choose from.
    llm_max_candidates: int = 20
    # Number of segments before/after the seed to include as context in the LLM window.
    # Ignored — window is now built by time range (min_clip_duration / 2 each side).
    llm_context_window: int = 2

    # Keywords for text scoring — broad set covering gaming, commentary, reactions
    keywords: list = field(default_factory=lambda: [
        # Excitement / reactions
        "crazy", "insane", "unbelievable", "incredible", "no way", "oh my god",
        "what the", "are you kidding", "seriously", "actually",
        # Gaming specific
        "watch this", "look at this", "did you see", "clutch", "let's go",
        "gg", "ez", "rip", "destroyed", "carried", "goat",
        # Emphasis
        "important", "wait", "hold on", "oh wow", "bro", "dude",
        "that was", "i can't believe", "never seen",
    ])

    # Clip selection
    top_n_clips: int = 6             # Raised from 5
    min_clip_duration: float = 30.0  # Raised from 20s — clips shorter than this feel too brief
    max_clip_duration: float = 60.0  # Raised from 45s to accommodate the longer minimum
    max_expansion_gap: float = 2.0   # Stop expanding across silence gaps > this many seconds
    # Minimum text score required for a segment to be considered for selection.
    # Segments below this threshold (and with no keywords) are skipped unless
    # there are not enough above-threshold candidates to fill top_n_clips.
    min_text_score_for_selection: float = 0.05

    # Subtitles
    burn_subtitles: bool = True      # Set to False to skip burning captions into clips
