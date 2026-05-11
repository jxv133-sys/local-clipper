# Creator Profile Module

This module provides persistence for creator profiles, which store metadata about content creators to calibrate clip selection scoring across multiple videos.

## Overview

Creator profiles are stored as JSON files in `~/.cache/local-clipper/profiles/{creator_id}.json` and contain:
- Content type (gaming, podcast, comedy, vlog, educational)
- Energy level (high, moderate, calm)
- Typical clip duration preference
- Creator-specific keyword overrides
- Video processing history

## Usage

### Creating a New Profile

```python
from pipeline.creator_profile import create_default_profile, save_creator_profile

# Create a gaming profile with auto-adjusted defaults
profile = create_default_profile("gaming_streamer_123", content_type="gaming")
# Result: energy_level="high", typical_clip_duration=35.0

# Create a podcast profile
profile = create_default_profile("podcast_host_456", content_type="podcast")
# Result: energy_level="calm", typical_clip_duration=60.0

# Save to disk
save_creator_profile(profile)
```

### Loading an Existing Profile

```python
from pipeline.creator_profile import load_creator_profile

# Load profile from default location
profile = load_creator_profile("gaming_streamer_123")

if profile:
    print(f"Content type: {profile.content_type}")
    print(f"Energy level: {profile.energy_level}")
    print(f"Videos processed: {profile.video_count}")
else:
    print("Profile not found")
```

### Updating a Profile

```python
from pipeline.creator_profile import load_creator_profile, save_creator_profile
from datetime import datetime, timezone

# Load existing profile
profile = load_creator_profile("gaming_streamer_123")

if profile:
    # Update fields
    profile.video_count += 1
    profile.keyword_overrides.append("new_keyword")
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    
    # Save changes
    save_creator_profile(profile)
```

### Custom Storage Location

```python
from pathlib import Path
from pipeline.creator_profile import load_creator_profile, save_creator_profile

custom_dir = Path("/custom/profiles/directory")

# Save to custom location
save_creator_profile(profile, profile_dir=custom_dir)

# Load from custom location
profile = load_creator_profile("creator_id", profile_dir=custom_dir)
```

## API Reference

### `load_creator_profile(creator_id: str, profile_dir: Optional[Path] = None) -> Optional[CreatorProfile]`

Load a creator profile from disk.

**Parameters:**
- `creator_id`: Unique identifier for the creator
- `profile_dir`: Optional custom directory (defaults to `~/.cache/local-clipper/profiles`)

**Returns:**
- `CreatorProfile` object if found, `None` if file doesn't exist or is invalid

**Error Handling:**
- Returns `None` for missing files (graceful handling)
- Returns `None` for invalid JSON or missing required fields
- Logs errors at ERROR level

### `save_creator_profile(profile: CreatorProfile, profile_dir: Optional[Path] = None) -> None`

Save a creator profile to disk.

**Parameters:**
- `profile`: CreatorProfile object to save
- `profile_dir`: Optional custom directory (defaults to `~/.cache/local-clipper/profiles`)

**Raises:**
- `OSError`: If directory creation or file writing fails

**Behavior:**
- Creates directory if it doesn't exist
- Overwrites existing profiles with the same creator_id
- Formats JSON with indent=2 for readability

### `create_default_profile(creator_id: str, content_type: str = "auto", energy_level: str = "moderate", typical_clip_duration: float = 45.0) -> CreatorProfile`

Create a new creator profile with sensible defaults.

**Parameters:**
- `creator_id`: Unique identifier for the creator
- `content_type`: Content category ("gaming", "podcast", "comedy", "vlog", "educational", "auto")
- `energy_level`: Typical energy level ("high", "moderate", "calm")
- `typical_clip_duration`: Preferred clip length in seconds

**Returns:**
- `CreatorProfile` object with default values and current timestamp

**Auto-Adjustments:**
- Gaming: `energy_level="high"`, `typical_clip_duration=35.0`
- Podcast: `energy_level="calm"`, `typical_clip_duration=60.0`
- Comedy: `energy_level="high"`, `typical_clip_duration=30.0`

## File Format

Profiles are stored as JSON files with the following structure:

```json
{
  "creator_id": "gaming_streamer_123",
  "content_type": "gaming",
  "energy_level": "high",
  "typical_clip_duration": 35.0,
  "keyword_overrides": ["clutch", "gg", "let's go"],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:45:00Z",
  "video_count": 12
}
```

## Integration

This module integrates with:
- `pipeline/models.py`: Uses the `CreatorProfile` dataclass
- `main.py`: Loads profiles at startup via `--creator-id` flag
- `scorer.py`: Uses profile data to calibrate LLM prompts
- `config.py`: Adjusts scoring weights based on energy level

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_creator_profile.py -v
```

Test coverage includes:
- Loading existing profiles
- Handling missing files gracefully
- Handling invalid JSON
- Saving new profiles
- Overwriting existing profiles
- Creating directories
- Default profile creation with auto-adjustments
- Round-trip persistence (save/load)
