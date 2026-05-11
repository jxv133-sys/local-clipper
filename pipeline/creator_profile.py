"""Creator profile persistence layer.

This module provides functions to load, save, and create creator profiles
that are stored in ~/.cache/local-clipper/profiles/{creator_id}.json.

Creator profiles store metadata about content creators to calibrate clip
selection scoring across multiple videos from the same creator.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from pipeline.models import CreatorProfile

logger = logging.getLogger(__name__)

# Default profile storage directory
DEFAULT_PROFILE_DIR = Path.home() / ".cache" / "local-clipper" / "profiles"


def load_creator_profile(creator_id: str, profile_dir: Optional[Path] = None) -> Optional[CreatorProfile]:
    """Load a creator profile from disk.
    
    Args:
        creator_id: Unique identifier for the creator (channel name or hash)
        profile_dir: Optional custom directory for profiles (defaults to ~/.cache/local-clipper/profiles)
    
    Returns:
        CreatorProfile object if found, None if file doesn't exist or is invalid
    
    Examples:
        >>> profile = load_creator_profile("gaming_streamer_123")
        >>> if profile:
        ...     print(f"Loaded profile for {profile.creator_id}")
    """
    if profile_dir is None:
        profile_dir = DEFAULT_PROFILE_DIR
    
    profile_path = profile_dir / f"{creator_id}.json"
    
    if not profile_path.exists():
        logger.debug(f"Profile not found: {profile_path}")
        return None
    
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        profile = CreatorProfile.from_dict(data)
        logger.info(f"Loaded creator profile: {creator_id} (content_type={profile.content_type}, video_count={profile.video_count})")
        return profile
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to load profile {profile_path}: {e}")
        return None


def save_creator_profile(profile: CreatorProfile, profile_dir: Optional[Path] = None) -> None:
    """Save a creator profile to disk.
    
    Creates the profile directory if it doesn't exist. Overwrites existing
    profile files with the same creator_id.
    
    Args:
        profile: CreatorProfile object to save
        profile_dir: Optional custom directory for profiles (defaults to ~/.cache/local-clipper/profiles)
    
    Raises:
        OSError: If directory creation or file writing fails
    
    Examples:
        >>> profile = CreatorProfile(
        ...     creator_id="gaming_streamer_123",
        ...     content_type="gaming",
        ...     energy_level="high",
        ...     typical_clip_duration=35.0,
        ...     keyword_overrides=["clutch", "gg"],
        ...     created_at="2024-01-15T10:30:00Z",
        ...     updated_at="2024-01-15T10:30:00Z",
        ...     video_count=1
        ... )
        >>> save_creator_profile(profile)
    """
    if profile_dir is None:
        profile_dir = DEFAULT_PROFILE_DIR
    
    # Create directory if it doesn't exist
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    profile_path = profile_dir / f"{profile.creator_id}.json"
    
    try:
        data = profile.to_dict()
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved creator profile: {profile.creator_id} to {profile_path}")
    
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save profile {profile_path}: {e}")
        raise


def create_default_profile(
    creator_id: str,
    content_type: str = "auto",
    energy_level: str = "moderate",
    typical_clip_duration: float = 45.0
) -> CreatorProfile:
    """Create a new creator profile with sensible defaults.
    
    Args:
        creator_id: Unique identifier for the creator
        content_type: Content category ("gaming", "podcast", "comedy", "vlog", "educational", "auto")
        energy_level: Typical energy level ("high", "moderate", "calm")
        typical_clip_duration: Preferred clip length in seconds (default: 45.0)
    
    Returns:
        CreatorProfile object with default values and current timestamp
    
    Examples:
        >>> profile = create_default_profile("new_creator", content_type="gaming", energy_level="high")
        >>> print(profile.creator_id)
        new_creator
    """
    from datetime import datetime, timezone
    
    # Generate ISO 8601 timestamp
    now = datetime.now(timezone.utc).isoformat()
    
    # Map content_type to sensible defaults
    if content_type == "auto":
        content_type = "vlog"  # Generic default
    
    # Adjust defaults based on content type
    if content_type == "gaming" and energy_level == "moderate":
        energy_level = "high"
        typical_clip_duration = 35.0
    elif content_type == "podcast" and energy_level == "moderate":
        energy_level = "calm"
        typical_clip_duration = 60.0
    elif content_type == "comedy" and energy_level == "moderate":
        energy_level = "high"
        typical_clip_duration = 30.0
    
    profile = CreatorProfile(
        creator_id=creator_id,
        content_type=content_type,
        energy_level=energy_level,
        typical_clip_duration=typical_clip_duration,
        keyword_overrides=[],
        created_at=now,
        updated_at=now,
        video_count=0
    )
    
    logger.info(f"Created default profile: {creator_id} (content_type={content_type}, energy_level={energy_level})")
    return profile
