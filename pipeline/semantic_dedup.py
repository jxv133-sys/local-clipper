"""Semantic deduplication using sentence-transformers embeddings.

This module provides semantic similarity computation between clip transcripts
using sentence-transformers (all-MiniLM-L6-v2 model). It goes beyond simple
word overlap (Jaccard similarity) to detect clips with similar meanings.

**Validates: Requirements 7.1**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.models import Clip, Transcript


def compute_semantic_similarity(
    clip_a: Clip,
    clip_b: Clip,
    transcript: Transcript,
) -> float:
    """Compute cosine similarity between clip embeddings.
    
    Strategy:
    1. Extract transcript text for each clip
    2. Encode with sentence-transformers (all-MiniLM-L6-v2)
    3. Compute cosine similarity
    
    Args:
        clip_a: First clip to compare
        clip_b: Second clip to compare
        transcript: Full transcript containing segment data
    
    Returns:
        Similarity score 0.0-1.0 (0.0 = completely different, 1.0 = identical)
    
    Raises:
        ImportError: If sentence-transformers is not installed
        RuntimeError: If model loading or encoding fails
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for semantic deduplication. "
            "Install with: pip install sentence-transformers"
        ) from e
    
    # Load model (cached after first call)
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        raise RuntimeError(f"Failed to load sentence-transformers model: {e}") from e
    
    # Extract transcript text for each clip
    text_a = _extract_clip_text(clip_a, transcript)
    text_b = _extract_clip_text(clip_b, transcript)
    
    # Handle empty text cases
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    # Encode texts to embeddings
    try:
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        embedding_a = embeddings[0]
        embedding_b = embeddings[1]
    except Exception as e:
        raise RuntimeError(f"Failed to encode text: {e}") from e
    
    # Compute cosine similarity
    # cosine_similarity = dot(a, b) / (norm(a) * norm(b))
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    # Avoid division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    
    # Clamp to [0.0, 1.0] range (cosine similarity can be negative for opposite vectors)
    similarity = max(0.0, min(1.0, float(similarity)))
    
    return similarity


def deduplicate_semantic(
    clips: list[Clip],
    transcript: Transcript,
    threshold: float = 0.8,
) -> list[Clip]:
    """Remove semantically similar clips.
    
    Strategy:
    1. Load sentence-transformers model (cached)
    2. Compute pairwise similarities
    3. For each pair above threshold, discard lower-scoring clip
    4. Return deduplicated list
    
    Args:
        clips: List of clips to deduplicate
        transcript: Full transcript containing segment data
        threshold: Similarity threshold (0.0-1.0). Pairs above this are considered duplicates.
                  Default 0.8 means clips must be very similar to be deduplicated.
    
    Returns:
        Filtered clip list with semantically similar clips removed
    
    Raises:
        ImportError: If sentence-transformers is not installed
        RuntimeError: If model loading or encoding fails
    
    **Validates: Requirements 7.2, 7.3**
    """
    if not clips:
        return clips
    
    if len(clips) == 1:
        return clips
    
    # Track which clips to keep (by index)
    clips_to_keep = set(range(len(clips)))
    
    # Compute pairwise similarities
    for i in range(len(clips)):
        if i not in clips_to_keep:
            continue  # Already discarded
        
        for j in range(i + 1, len(clips)):
            if j not in clips_to_keep:
                continue  # Already discarded
            
            # Compute similarity between clips[i] and clips[j]
            try:
                similarity = compute_semantic_similarity(
                    clips[i], clips[j], transcript
                )
            except (ImportError, RuntimeError) as e:
                # If similarity computation fails, re-raise to caller
                raise
            
            # If similarity above threshold, discard the lower-scoring clip
            if similarity > threshold:
                if clips[i].score >= clips[j].score:
                    # Keep clip i, discard clip j
                    clips_to_keep.discard(j)
                else:
                    # Keep clip j, discard clip i
                    clips_to_keep.discard(i)
                    break  # No need to compare clip i with remaining clips
    
    # Return deduplicated list (preserve original order)
    return [clips[i] for i in sorted(clips_to_keep)]


def _extract_clip_text(clip: Clip, transcript: Transcript) -> str:
    """Extract the full text content of a clip from the transcript.
    
    Args:
        clip: Clip with segment_indices
        transcript: Full transcript
    
    Returns:
        Concatenated text from all segments in the clip
    """
    texts = []
    for idx in clip.segment_indices:
        if 0 <= idx < len(transcript.segments):
            texts.append(transcript.segments[idx].text)
    
    return " ".join(texts)
