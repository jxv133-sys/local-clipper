from dataclasses import dataclass, field


@dataclass
class Config:
    # Paths
    work_dir: str                    # Temporary working directory (set at runtime)
    output_dir: str = "output"       # Where final clips are saved

    # Whisper
    whisper_model: str = "base"      # Whisper model size: tiny/base/small/medium/large

    # Scoring weights
    text_weight: float = 0.4
    audio_weight: float = 0.6
    llm_weight: float = 0.0          # 0.0 = LLM disabled

    # LLM (optional)
    llm_enabled: bool = False
    llm_endpoint: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3"

    # Keywords for text scoring
    keywords: list = field(default_factory=lambda: [
        "crazy", "important", "watch this", "incredible", "unbelievable"
    ])

    # Clip selection
    top_n_clips: int = 5
    min_clip_duration: float = 20.0
    max_clip_duration: float = 45.0
