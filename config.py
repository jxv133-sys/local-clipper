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
    min_clip_duration: float = 20.0
    max_clip_duration: float = 45.0
