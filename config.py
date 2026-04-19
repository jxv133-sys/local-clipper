import os
from dataclasses import dataclass, field


def _default_llm_endpoint() -> str:
    host = os.getenv("OLLAMA_HOST", "").strip()
    if host:
        host = host.rstrip("/")
        if host.endswith("/api/generate"):
            return host
        return f"{host}/api/generate"
    return "http://localhost:11434/api/generate"


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

    # LLM pre-filter weights — used by _build_candidate_windows to rank candidates
    # before sending them to the LLM.  Audio-heavy so energetic silent moments
    # surface even when their text score is low.
    # These are intentionally separate from text_weight/audio_weight, which are
    # used for the final clip_score combination after LLM scoring.
    llm_prefilter_text_weight: float = 0.2
    llm_prefilter_audio_weight: float = 0.8

    # LLM (optional)
    llm_enabled: bool = False
    llm_endpoint: str = field(default_factory=_default_llm_endpoint)
    llm_model: str = "llama3"
    # Maximum number of candidate windows sent to the LLM.
    # The scorer picks the top-scoring moments (spaced >= min_clip_duration apart)
    # and calls the LLM once per window — never once per Whisper segment.
    # Rule of thumb: top_n_clips * 3 gives the LLM enough candidates to choose from.
    llm_max_candidates: int = 20
    # Number of pure-audio (spike-score) slots reserved in the LLM candidate list.
    # These slots are filled by the top segments ranked by spike_score alone,
    # guaranteeing that high-energy silent moments always reach LLM scoring.
    # The remaining (llm_max_candidates - llm_audio_candidates) slots are filled
    # by the text+audio combined pre-score (current behaviour).
    # Set to 0 to disable the audio-only track entirely.
    llm_audio_candidates: int = 10
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

    # Single-word reaction exclamations that Whisper reliably transcribes from brief outbursts.
    # Matched as whole words (word-boundary aware) and case-insensitively.
    # Weighted higher than regular keywords because they indicate a pure reaction moment.
    reaction_keywords: list = field(default_factory=lambda: [
        "oh", "wow", "whoa", "no", "yes", "what", "ahhh", "omg", "noo", "yoo", "bro",
        "wait", "stop", "go", "run", "help", "dead", "gone", "hit", "fly", "fall",
    ])

    # Score added per occurrence of a reaction keyword (vs 2.0 for regular keywords).
    reaction_weight: float = 3.0

    # Clip selection
    top_n_clips: int = 6             # Raised from 5
    min_clip_duration: float = 30.0  # Raised from 20s — clips shorter than this feel too brief
    max_clip_duration: float = 60.0  # Raised from 45s to accommodate the longer minimum
    max_expansion_gap: float = 15.0  # Stop expanding across silence gaps > this many seconds
    # Minimum text score required for a segment to be considered for selection.
    # Segments below this threshold (and with no keywords) are skipped unless
    # there are not enough above-threshold candidates to fill top_n_clips.
    min_text_score_for_selection: float = 0.05

    # Spike detection weight — contribution of audio spike score to clip_score
    # A spike score detects sudden bursts of audio energy relative to a rolling baseline.
    # Set to 0.0 to disable spike detection entirely.
    spike_weight: float = 0.2

    # Burst detection weight — contribution of silence-then-burst score to clip_score.
    # A burst score detects the "silence → loud" transition pattern: a segment that is
    # preceded by near-silence (< 10% of global mean RMS) and is itself loud
    # (> 50% of global max RMS).  Binary: 1.0 if the pattern matches, 0.0 otherwise.
    # Set to 0.0 to disable burst detection entirely.
    burst_weight: float = 0.3

    # Subtitles
    burn_subtitles: bool = True      # Set to False to skip burning captions into clips
