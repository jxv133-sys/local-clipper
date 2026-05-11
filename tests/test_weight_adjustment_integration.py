"""Integration test for creator profile weight adjustment in main.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from config import Config
from pipeline.creator_profile import create_default_profile, save_creator_profile
from pipeline.models import CreatorProfile


def test_weight_adjustment_integration_high_energy():
    """Test that high-energy profile adjusts weights correctly in build_config flow."""
    # Create a temporary profile directory
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_dir = Path(tmpdir)
        
        # Create and save a high-energy profile
        profile = create_default_profile("test_gaming_creator", content_type="gaming", energy_level="high")
        save_creator_profile(profile, profile_dir)
        
        # Create config and simulate the build_config flow
        cfg = Config(work_dir="/tmp/test")
        cfg.creator_id = "test_gaming_creator"
        cfg.creator_profile_path = str(profile_dir)
        
        # Load the profile (simulating what main.py does)
        from pipeline.creator_profile import load_creator_profile
        loaded_profile = load_creator_profile("test_gaming_creator", profile_dir)
        cfg.creator_profile = loaded_profile
        
        # Adjust weights (simulating what main.py does)
        cfg.adjust_weights_for_creator_profile()
        
        # Verify high-energy adjustment: 40% text, 60% audio
        assert cfg.text_weight == pytest.approx(0.40)
        assert cfg.audio_weight == pytest.approx(0.60)


def test_weight_adjustment_integration_calm():
    """Test that calm profile adjusts weights correctly in build_config flow."""
    # Create a temporary profile directory
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_dir = Path(tmpdir)
        
        # Create and save a calm profile
        profile = create_default_profile("test_podcast_creator", content_type="podcast", energy_level="calm")
        save_creator_profile(profile, profile_dir)
        
        # Create config and simulate the build_config flow
        cfg = Config(work_dir="/tmp/test")
        cfg.creator_id = "test_podcast_creator"
        cfg.creator_profile_path = str(profile_dir)
        
        # Load the profile (simulating what main.py does)
        from pipeline.creator_profile import load_creator_profile
        loaded_profile = load_creator_profile("test_podcast_creator", profile_dir)
        cfg.creator_profile = loaded_profile
        
        # Adjust weights (simulating what main.py does)
        cfg.adjust_weights_for_creator_profile()
        
        # Verify calm adjustment: 60% text, 40% audio
        assert cfg.text_weight == pytest.approx(0.60)
        assert cfg.audio_weight == pytest.approx(0.40)


def test_weight_adjustment_integration_with_llm():
    """Test that weight adjustment works correctly when LLM is enabled."""
    # Create a temporary profile directory
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_dir = Path(tmpdir)
        
        # Create and save a high-energy profile
        profile = create_default_profile("test_gaming_creator", content_type="gaming", energy_level="high")
        save_creator_profile(profile, profile_dir)
        
        # Create config with LLM enabled
        cfg = Config(work_dir="/tmp/test", llm_enabled=True, text_weight=0.35, audio_weight=0.25, llm_weight=0.4)
        cfg.creator_id = "test_gaming_creator"
        cfg.creator_profile_path = str(profile_dir)
        
        # Load the profile
        from pipeline.creator_profile import load_creator_profile
        loaded_profile = load_creator_profile("test_gaming_creator", profile_dir)
        cfg.creator_profile = loaded_profile
        
        # Adjust weights
        cfg.adjust_weights_for_creator_profile()
        
        # Non-LLM budget = 0.35 + 0.25 = 0.6
        # High-energy: 40% of 0.6 = 0.24 text, 60% of 0.6 = 0.36 audio
        assert cfg.text_weight == pytest.approx(0.24)
        assert cfg.audio_weight == pytest.approx(0.36)
        assert cfg.llm_weight == pytest.approx(0.4)
        
        # Total should sum to 1.0
        assert cfg.text_weight + cfg.audio_weight + cfg.llm_weight == pytest.approx(1.0)
