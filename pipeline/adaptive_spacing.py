"""Adaptive spacing constraint computation for clip selection.

This module provides functionality to dynamically adjust spacing constraints
based on video duration and the number of clips requested, ensuring optimal
clip distribution across videos of varying lengths.

**Validates: Requirements 8.1, 8.2**
"""


def compute_adaptive_spacing(
    video_duration: float,
    top_n_clips: int,
    base_spacing: float,
) -> float:
    """Compute effective spacing constraint based on video duration and clip count.
    
    The adaptive spacing algorithm ensures that clips can be distributed evenly
    across the video while respecting a minimum floor to prevent over-clustering.
    
    Strategy:
    1. Calculate required spacing: video_duration / (top_n_clips + 1)
    2. Take the minimum of base_spacing and required spacing
    3. Apply minimum floor (30.0 seconds) to prevent over-clustering
    
    Formula:
        effective = max(30.0, min(base_spacing, video_duration / (top_n_clips + 1)))
    
    Args:
        video_duration: Total duration of the video in seconds
        top_n_clips: Number of clips to select
        base_spacing: Base spacing constraint in seconds (from config)
    
    Returns:
        Effective spacing constraint in seconds
    
    Examples:
        >>> compute_adaptive_spacing(600.0, 6, 300.0)  # 10 min video, 6 clips
        85.71428571428571
        
        >>> compute_adaptive_spacing(1800.0, 6, 300.0)  # 30 min video, 6 clips
        257.14285714285717
        
        >>> compute_adaptive_spacing(3600.0, 6, 300.0)  # 60 min video, 6 clips
        300.0
        
        >>> compute_adaptive_spacing(300.0, 6, 300.0)  # 5 min video, 6 clips
        42.857142857142854
        
        >>> compute_adaptive_spacing(180.0, 10, 300.0)  # 3 min video, 10 clips (edge case)
        30.0
    """
    min_floor = 30.0  # Minimum spacing to prevent over-clustering
    
    # Calculate required spacing to fit all clips
    required_spacing = video_duration / (top_n_clips + 1)
    
    # Apply formula: max(min_floor, min(base_spacing, required_spacing))
    effective_spacing = max(min_floor, min(base_spacing, required_spacing))
    
    return effective_spacing
