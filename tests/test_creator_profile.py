"""Unit tests for creator_profile.py module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.creator_profile import (
    create_default_profile,
    load_creator_profile,
    save_creator_profile,
)
from pipeline.models import CreatorProfile


class TestLoadCreatorProfile:
    """Tests for load_creator_profile function."""
    
    def test_load_existing_profile(self, tmp_path: Path):
        """Test loading a valid profile from disk."""
        # Create a test profile file
        profile_data = {
            "creator_id": "test_creator",
            "content_type": "gaming",
            "energy_level": "high",
            "typical_clip_duration": 35.0,
            "keyword_overrides": ["clutch", "gg"],
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-20T14:45:00Z",
            "video_count": 12
        }
        
        profile_path = tmp_path / "test_creator.json"
        with open(profile_path, "w") as f:
            json.dump(profile_data, f)
        
        # Load the profile
        profile = load_creator_profile("test_creator", profile_dir=tmp_path)
        
        assert profile is not None
        assert profile.creator_id == "test_creator"
        assert profile.content_type == "gaming"
        assert profile.energy_level == "high"
        assert profile.typical_clip_duration == 35.0
        assert profile.keyword_overrides == ["clutch", "gg"]
        assert profile.video_count == 12
    
    def test_load_nonexistent_profile(self, tmp_path: Path):
        """Test loading a profile that doesn't exist returns None."""
        profile = load_creator_profile("nonexistent_creator", profile_dir=tmp_path)
        assert profile is None
    
    def test_load_invalid_json(self, tmp_path: Path):
        """Test loading a profile with invalid JSON returns None."""
        profile_path = tmp_path / "invalid_creator.json"
        with open(profile_path, "w") as f:
            f.write("{ invalid json }")
        
        profile = load_creator_profile("invalid_creator", profile_dir=tmp_path)
        assert profile is None
    
    def test_load_missing_required_field(self, tmp_path: Path):
        """Test loading a profile with missing required fields returns None."""
        profile_data = {
            "creator_id": "incomplete_creator",
            "content_type": "gaming",
            # Missing other required fields
        }
        
        profile_path = tmp_path / "incomplete_creator.json"
        with open(profile_path, "w") as f:
            json.dump(profile_data, f)
        
        profile = load_creator_profile("incomplete_creator", profile_dir=tmp_path)
        assert profile is None


class TestSaveCreatorProfile:
    """Tests for save_creator_profile function."""
    
    def test_save_new_profile(self, tmp_path: Path):
        """Test saving a new profile creates the file."""
        profile = CreatorProfile(
            creator_id="new_creator",
            content_type="podcast",
            energy_level="calm",
            typical_clip_duration=60.0,
            keyword_overrides=["interesting", "fascinating"],
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
            video_count=1
        )
        
        save_creator_profile(profile, profile_dir=tmp_path)
        
        # Verify file was created
        profile_path = tmp_path / "new_creator.json"
        assert profile_path.exists()
        
        # Verify content
        with open(profile_path, "r") as f:
            data = json.load(f)
        
        assert data["creator_id"] == "new_creator"
        assert data["content_type"] == "podcast"
        assert data["energy_level"] == "calm"
        assert data["typical_clip_duration"] == 60.0
        assert data["keyword_overrides"] == ["interesting", "fascinating"]
        assert data["video_count"] == 1
    
    def test_save_overwrites_existing_profile(self, tmp_path: Path):
        """Test saving a profile overwrites existing file."""
        # Create initial profile
        profile1 = CreatorProfile(
            creator_id="test_creator",
            content_type="gaming",
            energy_level="high",
            typical_clip_duration=35.0,
            keyword_overrides=["old"],
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
            video_count=1
        )
        save_creator_profile(profile1, profile_dir=tmp_path)
        
        # Update and save again
        profile2 = CreatorProfile(
            creator_id="test_creator",
            content_type="gaming",
            energy_level="high",
            typical_clip_duration=35.0,
            keyword_overrides=["new"],
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-20T14:45:00Z",
            video_count=5
        )
        save_creator_profile(profile2, profile_dir=tmp_path)
        
        # Verify updated content
        profile_path = tmp_path / "test_creator.json"
        with open(profile_path, "r") as f:
            data = json.load(f)
        
        assert data["keyword_overrides"] == ["new"]
        assert data["video_count"] == 5
        assert data["updated_at"] == "2024-01-20T14:45:00Z"
    
    def test_save_creates_directory(self, tmp_path: Path):
        """Test saving a profile creates the directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "profiles"
        
        profile = CreatorProfile(
            creator_id="test_creator",
            content_type="vlog",
            energy_level="moderate",
            typical_clip_duration=45.0,
            keyword_overrides=[],
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T10:30:00Z",
            video_count=0
        )
        
        save_creator_profile(profile, profile_dir=nested_dir)
        
        # Verify directory and file were created
        assert nested_dir.exists()
        assert (nested_dir / "test_creator.json").exists()


class TestCreateDefaultProfile:
    """Tests for create_default_profile function."""
    
    def test_create_default_profile_basic(self):
        """Test creating a default profile with minimal arguments."""
        profile = create_default_profile("test_creator")
        
        assert profile.creator_id == "test_creator"
        assert profile.content_type == "vlog"  # Default when "auto"
        assert profile.energy_level == "moderate"
        assert profile.typical_clip_duration == 45.0
        assert profile.keyword_overrides == []
        assert profile.video_count == 0
        assert profile.created_at == profile.updated_at
    
    def test_create_default_profile_gaming(self):
        """Test creating a gaming profile adjusts defaults."""
        profile = create_default_profile("gaming_creator", content_type="gaming")
        
        assert profile.content_type == "gaming"
        assert profile.energy_level == "high"  # Auto-adjusted for gaming
        assert profile.typical_clip_duration == 35.0  # Shorter for gaming
    
    def test_create_default_profile_podcast(self):
        """Test creating a podcast profile adjusts defaults."""
        profile = create_default_profile("podcast_creator", content_type="podcast")
        
        assert profile.content_type == "podcast"
        assert profile.energy_level == "calm"  # Auto-adjusted for podcast
        assert profile.typical_clip_duration == 60.0  # Longer for podcast
    
    def test_create_default_profile_comedy(self):
        """Test creating a comedy profile adjusts defaults."""
        profile = create_default_profile("comedy_creator", content_type="comedy")
        
        assert profile.content_type == "comedy"
        assert profile.energy_level == "high"  # Auto-adjusted for comedy
        assert profile.typical_clip_duration == 30.0  # Shorter for comedy
    
    def test_create_default_profile_custom_energy(self):
        """Test creating a profile with custom energy level."""
        profile = create_default_profile(
            "custom_creator",
            content_type="gaming",
            energy_level="calm"  # Override default
        )
        
        assert profile.energy_level == "calm"  # Should not be auto-adjusted
    
    def test_create_default_profile_custom_duration(self):
        """Test creating a profile with custom clip duration."""
        profile = create_default_profile(
            "custom_creator",
            typical_clip_duration=90.0
        )
        
        assert profile.typical_clip_duration == 90.0
    
    def test_create_default_profile_timestamps(self):
        """Test that created_at and updated_at are valid ISO 8601 timestamps."""
        profile = create_default_profile("test_creator")
        
        # Verify timestamps are valid ISO 8601
        created_dt = datetime.fromisoformat(profile.created_at.replace("Z", "+00:00"))
        updated_dt = datetime.fromisoformat(profile.updated_at.replace("Z", "+00:00"))
        
        assert created_dt.tzinfo is not None  # Has timezone
        assert updated_dt.tzinfo is not None
        assert created_dt == updated_dt  # Should be the same for new profiles


class TestRoundTripPersistence:
    """Tests for save/load round-trip behavior."""
    
    def test_save_and_load_round_trip(self, tmp_path: Path):
        """Test that saving and loading a profile preserves all data."""
        original = CreatorProfile(
            creator_id="round_trip_creator",
            content_type="educational",
            energy_level="moderate",
            typical_clip_duration=50.0,
            keyword_overrides=["learn", "understand", "concept"],
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-20T14:45:00Z",
            video_count=7
        )
        
        # Save and load
        save_creator_profile(original, profile_dir=tmp_path)
        loaded = load_creator_profile("round_trip_creator", profile_dir=tmp_path)
        
        # Verify all fields match
        assert loaded is not None
        assert loaded.creator_id == original.creator_id
        assert loaded.content_type == original.content_type
        assert loaded.energy_level == original.energy_level
        assert loaded.typical_clip_duration == original.typical_clip_duration
        assert loaded.keyword_overrides == original.keyword_overrides
        assert loaded.created_at == original.created_at
        assert loaded.updated_at == original.updated_at
        assert loaded.video_count == original.video_count
    
    def test_create_save_load_round_trip(self, tmp_path: Path):
        """Test that creating, saving, and loading a profile works."""
        # Create default profile
        created = create_default_profile("new_creator", content_type="gaming")
        
        # Save it
        save_creator_profile(created, profile_dir=tmp_path)
        
        # Load it back
        loaded = load_creator_profile("new_creator", profile_dir=tmp_path)
        
        # Verify all fields match
        assert loaded is not None
        assert loaded.creator_id == created.creator_id
        assert loaded.content_type == created.content_type
        assert loaded.energy_level == created.energy_level
        assert loaded.typical_clip_duration == created.typical_clip_duration
        assert loaded.keyword_overrides == created.keyword_overrides
        assert loaded.created_at == created.created_at
        assert loaded.updated_at == created.updated_at
        assert loaded.video_count == created.video_count
