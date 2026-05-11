"""Property-based tests for clip-selection-improvements feature.

This module contains property-based tests that validate correctness properties
defined in .kiro/specs/clip-selection-improvements/design.md.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from config import Config
from pipeline.models import CreatorProfile


# ---------------------------------------------------------------------------
# Hypothesis strategies for Config validation
# ---------------------------------------------------------------------------

@st.composite
def invalid_weight_sum_config(draw):
    """Generate Config parameters where text_weight + audio_weight + llm_weight != 1.0.
    
    Returns a tuple of (text_weight, audio_weight, llm_weight) that don't sum to 1.0.
    """
    # Generate three weights that don't sum to 1.0
    text_weight = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    audio_weight = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    llm_weight = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Ensure they don't sum to 1.0 (outside tolerance)
    weight_sum = text_weight + audio_weight + llm_weight
    assume(abs(weight_sum - 1.0) > 1e-9)
    
    return text_weight, audio_weight, llm_weight


@st.composite
def invalid_duration_config(draw):
    """Generate Config parameters where min_clip_duration > max_clip_duration.
    
    Returns a tuple of (min_clip_duration, max_clip_duration) where min > max.
    """
    max_duration = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    min_duration = draw(st.floats(min_value=max_duration + 0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
    
    return min_duration, max_duration


# ---------------------------------------------------------------------------
# Hypothesis strategies for CreatorProfile
# ---------------------------------------------------------------------------

@st.composite
def creator_profile_strategy(draw):
    """Generate valid CreatorProfile instances.
    
    Returns a CreatorProfile with all valid field values.
    """
    creator_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    content_type = draw(st.sampled_from(["gaming", "podcast", "comedy", "vlog", "educational"]))
    energy_level = draw(st.sampled_from(["high", "moderate", "calm"]))
    typical_clip_duration = draw(st.floats(min_value=10.0, max_value=120.0, allow_nan=False, allow_infinity=False))
    keyword_overrides = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        min_size=0,
        max_size=10
    ))
    
    # Generate ISO 8601 timestamps
    created_at = draw(st.datetimes(
        min_value=__import__('datetime').datetime(2020, 1, 1),
        max_value=__import__('datetime').datetime(2025, 12, 31)
    )).isoformat() + "Z"
    
    updated_at = draw(st.datetimes(
        min_value=__import__('datetime').datetime(2020, 1, 1),
        max_value=__import__('datetime').datetime(2025, 12, 31)
    )).isoformat() + "Z"
    
    video_count = draw(st.integers(min_value=0, max_value=10000))
    
    return CreatorProfile(
        creator_id=creator_id,
        content_type=content_type,
        energy_level=energy_level,
        typical_clip_duration=typical_clip_duration,
        keyword_overrides=keyword_overrides,
        created_at=created_at,
        updated_at=updated_at,
        video_count=video_count,
    )


# ---------------------------------------------------------------------------
# Property 14: Config Validation Constraints
# Validates: Requirements 16.1, 16.3, 16.4
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 14: Config Validation Constraints
@given(weights=invalid_weight_sum_config())
@settings(max_examples=100)
def test_config_invalid_weight_sum_raises_error(weights):
    """For any Config with llm_enabled=True where text_weight + audio_weight + llm_weight != 1.0,
    initialization should raise a descriptive ValueError.
    
    **Validates: Requirements 16.1, 16.3**
    """
    text_weight, audio_weight, llm_weight = weights
    
    with pytest.raises(ValueError, match="text_weight \\+ audio_weight \\+ llm_weight"):
        Config(
            work_dir="/tmp/test",
            llm_enabled=True,
            text_weight=text_weight,
            audio_weight=audio_weight,
            llm_weight=llm_weight,
        )


# Feature: clip-selection-improvements, Property 14: Config Validation Constraints
@given(durations=invalid_duration_config())
@settings(max_examples=100)
def test_config_invalid_duration_ordering_raises_error(durations):
    """For any Config where min_clip_duration > max_clip_duration,
    initialization should raise a descriptive ValueError.
    
    **Validates: Requirements 16.4**
    """
    min_duration, max_duration = durations
    
    with pytest.raises(ValueError, match="min_clip_duration"):
        Config(
            work_dir="/tmp/test",
            min_clip_duration=min_duration,
            max_clip_duration=max_duration,
        )


# Feature: clip-selection-improvements, Property 14: Config Validation Constraints
@given(
    text_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    audio_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_config_valid_weight_sum_succeeds(text_weight, audio_weight):
    """For any Config with llm_enabled=True where text_weight + audio_weight + llm_weight == 1.0,
    initialization should succeed without raising an error.
    
    **Validates: Requirements 16.1, 16.3**
    """
    # Calculate llm_weight to make the sum exactly 1.0
    llm_weight = 1.0 - text_weight - audio_weight
    
    # Only test when llm_weight is valid (non-negative)
    assume(llm_weight >= 0.0)
    
    # Should not raise
    config = Config(
        work_dir="/tmp/test",
        llm_enabled=True,
        text_weight=text_weight,
        audio_weight=audio_weight,
        llm_weight=llm_weight,
    )
    
    # Verify the weights are set correctly
    assert config.text_weight == pytest.approx(text_weight)
    assert config.audio_weight == pytest.approx(audio_weight)
    assert config.llm_weight == pytest.approx(llm_weight)
    assert abs(config.text_weight + config.audio_weight + config.llm_weight - 1.0) <= 1e-9


# Feature: clip-selection-improvements, Property 14: Config Validation Constraints
@given(
    min_duration=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    max_duration=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_config_valid_duration_ordering_succeeds(min_duration, max_duration):
    """For any Config where min_clip_duration <= max_clip_duration,
    initialization should succeed without raising an error.
    
    **Validates: Requirements 16.4**
    """
    # Only test when min <= max
    assume(min_duration <= max_duration)
    
    # Should not raise
    config = Config(
        work_dir="/tmp/test",
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
    )
    
    # Verify the durations are set correctly
    assert config.min_clip_duration == pytest.approx(min_duration)
    assert config.max_clip_duration == pytest.approx(max_duration)
    assert config.min_clip_duration <= config.max_clip_duration


# ---------------------------------------------------------------------------
# Property 1: Profile Field Persistence
# Validates: Requirements 1.2, 1.3, 1.4, 15.6
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 1: Profile Field Persistence
@given(profile=creator_profile_strategy())
@settings(max_examples=100)
def test_profile_field_persistence(profile):
    """For any valid CreatorProfile, all fields should be preserved with correct types and values.
    
    This test validates that CreatorProfile instances maintain data integrity:
    - content_type is one of the valid enum values
    - energy_level is one of the valid enum values
    - typical_clip_duration is positive
    - keyword_overrides is a list
    - video_count is non-negative
    
    **Validates: Requirements 1.2, 1.3, 1.4, 15.6**
    """
    # Validate content_type field
    assert profile.content_type in ["gaming", "podcast", "comedy", "vlog", "educational"], \
        f"content_type must be valid enum value, got: {profile.content_type}"
    
    # Validate energy_level field
    assert profile.energy_level in ["high", "moderate", "calm"], \
        f"energy_level must be valid enum value, got: {profile.energy_level}"
    
    # Validate typical_clip_duration is positive
    assert profile.typical_clip_duration > 0, \
        f"typical_clip_duration must be positive, got: {profile.typical_clip_duration}"
    
    # Validate keyword_overrides is a list
    assert isinstance(profile.keyword_overrides, list), \
        f"keyword_overrides must be a list, got: {type(profile.keyword_overrides)}"
    
    # Validate all keywords are strings
    for keyword in profile.keyword_overrides:
        assert isinstance(keyword, str), \
            f"All keyword_overrides must be strings, got: {type(keyword)}"
    
    # Validate video_count is non-negative
    assert profile.video_count >= 0, \
        f"video_count must be non-negative, got: {profile.video_count}"
    
    # Validate creator_id is non-empty
    assert len(profile.creator_id) > 0, \
        "creator_id must be non-empty"
    
    # Validate timestamps are strings (ISO 8601 format)
    assert isinstance(profile.created_at, str), \
        f"created_at must be a string, got: {type(profile.created_at)}"
    assert isinstance(profile.updated_at, str), \
        f"updated_at must be a string, got: {type(profile.updated_at)}"


# ---------------------------------------------------------------------------
# Property 2: Creator Profile Round-Trip Serialization
# Validates: Requirements 1.2, 1.3, 1.4, 15.6
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 2: Creator Profile Round-Trip Serialization
@given(profile=creator_profile_strategy())
@settings(max_examples=100)
def test_creator_profile_round_trip_serialization(profile):
    """For any valid CreatorProfile, to_dict → from_dict produces equivalent object.
    
    This test validates the round-trip serialization property:
    1. Serialize a CreatorProfile to a dict using to_dict()
    2. Deserialize the dict back to a CreatorProfile using from_dict()
    3. Verify all fields match the original
    
    This ensures that CreatorProfile persistence to disk (JSON) works correctly
    and no data is lost or corrupted during serialization/deserialization.
    
    **Validates: Requirements 1.2, 1.3, 1.4, 15.6**
    """
    # Serialize to dict
    profile_dict = profile.to_dict()
    
    # Verify dict has all required keys
    expected_keys = {
        "creator_id", "content_type", "energy_level", "typical_clip_duration",
        "keyword_overrides", "created_at", "updated_at", "video_count"
    }
    assert set(profile_dict.keys()) == expected_keys, \
        f"Serialized dict missing keys. Expected: {expected_keys}, Got: {set(profile_dict.keys())}"
    
    # Deserialize from dict
    parsed_profile = CreatorProfile.from_dict(profile_dict)
    
    # Verify all fields match
    assert parsed_profile.creator_id == profile.creator_id, \
        f"creator_id mismatch: {parsed_profile.creator_id} != {profile.creator_id}"
    
    assert parsed_profile.content_type == profile.content_type, \
        f"content_type mismatch: {parsed_profile.content_type} != {profile.content_type}"
    
    assert parsed_profile.energy_level == profile.energy_level, \
        f"energy_level mismatch: {parsed_profile.energy_level} != {profile.energy_level}"
    
    assert parsed_profile.typical_clip_duration == pytest.approx(profile.typical_clip_duration), \
        f"typical_clip_duration mismatch: {parsed_profile.typical_clip_duration} != {profile.typical_clip_duration}"
    
    assert parsed_profile.keyword_overrides == profile.keyword_overrides, \
        f"keyword_overrides mismatch: {parsed_profile.keyword_overrides} != {profile.keyword_overrides}"
    
    assert parsed_profile.created_at == profile.created_at, \
        f"created_at mismatch: {parsed_profile.created_at} != {profile.created_at}"
    
    assert parsed_profile.updated_at == profile.updated_at, \
        f"updated_at mismatch: {parsed_profile.updated_at} != {profile.updated_at}"
    
    assert parsed_profile.video_count == profile.video_count, \
        f"video_count mismatch: {parsed_profile.video_count} != {profile.video_count}"
    
    # Verify round-trip can be repeated (idempotent)
    second_dict = parsed_profile.to_dict()
    assert second_dict == profile_dict, \
        "Second serialization should produce identical dict"


# ---------------------------------------------------------------------------
# Property 4: Phrase Weight Superiority
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

@st.composite
def text_with_phrase_strategy(draw):
    """Generate text containing a phrase from the phrase list.
    
    Returns a tuple of (text, phrase) where text contains the phrase.
    """
    # Common phrases from default config
    phrases = ["oh my god", "no way", "watch this", "look at this"]
    phrase = draw(st.sampled_from(phrases))
    
    # Generate prefix and suffix that don't contain the phrase words
    # to avoid accidental individual word matches
    safe_words = ["that", "was", "really", "amazing", "incredible", "cool", "great"]
    prefix_words = draw(st.lists(st.sampled_from(safe_words), min_size=0, max_size=3))
    suffix_words = draw(st.lists(st.sampled_from(safe_words), min_size=0, max_size=3))
    
    prefix = " ".join(prefix_words) + (" " if prefix_words else "")
    suffix = (" " if suffix_words else "") + " ".join(suffix_words)
    
    text = prefix + phrase + suffix
    return text.strip(), phrase


# Feature: clip-selection-improvements, Property 4: Phrase Weight Superiority
@given(
    phrase=st.sampled_from(["oh my god", "no way", "watch this", "look at this"]),
    prefix=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), whitelist_characters=' '), min_size=0, max_size=20),
    suffix=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), whitelist_characters=' '), min_size=0, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_phrase_weight_superiority(phrase, prefix, suffix):
    """For any text containing a phrase, the score with phrase detection enabled
    should be higher than the score with only individual word keywords.
    
    This test validates Requirement 4.3: phrase_weight should provide a scoring
    advantage over individual keyword matches.
    
    The test compares the SAME text scored two ways:
    1. With phrase_keywords=[phrase] + keywords=[phrase words] (phrase + words)
    2. With phrase_keywords=[] + keywords=[phrase words] (words only)
    
    The first configuration should always score higher because it gets both the
    phrase bonus (4.0) AND the individual word matches (2.0 each), while the
    second only gets the individual word matches.
    
    **Validates: Requirements 4.3**
    """
    from pipeline.models import Segment
    from pipeline.scorer import compute_text_score
    
    # Build text with phrase
    text = f"{prefix} {phrase} {suffix}".strip()
    
    # Skip if text is empty or doesn't contain the phrase
    assume(len(text) > 0)
    assume(phrase.lower() in text.lower())
    
    # Config 1: Phrase detection enabled, phrase words also as regular keywords
    # This simulates the real-world scenario where both are active
    config_with_phrase = Config(work_dir="/tmp/test")
    config_with_phrase.phrase_keywords = [phrase]
    config_with_phrase.phrase_weight = 4.0
    config_with_phrase.keywords = phrase.split()  # Individual words also present
    config_with_phrase.reaction_keywords = []
    config_with_phrase.text_pattern_weight = 0.0
    
    # Config 2: Only individual word keywords (no phrase detection)
    config_without_phrase = Config(work_dir="/tmp/test")
    config_without_phrase.phrase_keywords = []
    config_without_phrase.keywords = phrase.split()
    config_without_phrase.reaction_keywords = []
    config_without_phrase.text_pattern_weight = 0.0
    
    # Create segment
    segment = Segment(start=0.0, end=1.0, text=text)
    
    # Compute scores
    score_with_phrase = compute_text_score(config_with_phrase, segment)
    score_without_phrase = compute_text_score(config_without_phrase, segment)
    
    # Score with phrase detection should be higher (gets phrase bonus + word matches)
    # vs just word matches alone
    assert score_with_phrase > score_without_phrase, \
        f"Score with phrase detection ({score_with_phrase:.4f}) should be higher than " \
        f"score without phrase detection ({score_without_phrase:.4f}) for text: '{text}'"
