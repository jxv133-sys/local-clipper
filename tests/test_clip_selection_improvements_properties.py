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
@settings(max_examples=20)
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
@settings(max_examples=20)
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
@settings(max_examples=20)
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
@settings(max_examples=20)
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
@settings(max_examples=20)
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
@settings(max_examples=20)
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
@settings(max_examples=20, deadline=None)
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


# ---------------------------------------------------------------------------
# Property 15: Video Summary Sampling Rate
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@st.composite
def transcript_strategy(draw):
    """Generate valid Transcript instances with varying numbers of segments.
    
    Returns a Transcript with N segments where N is between 1 and 200.
    """
    num_segments = draw(st.integers(min_value=1, max_value=200))
    
    segments = []
    current_time = 0.0
    
    for i in range(num_segments):
        # Each segment is 3-10 seconds long
        duration = draw(st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        start = current_time
        end = current_time + duration
        
        # Generate simple text for the segment (shorter to reduce entropy)
        text = draw(st.text(
            alphabet='abcdefghijklmnopqrstuvwxyz ',
            min_size=10,
            max_size=50
        ))
        
        from pipeline.models import Segment
        segments.append(Segment(start=start, end=end, text=text))
        current_time = end
    
    from pipeline.models import Transcript
    return Transcript(segments=segments)


# Feature: clip-selection-improvements, Property 15: Video Summary Sampling Rate
@given(
    transcript=transcript_strategy(),
    sample_rate=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=20, deadline=None)
def test_video_summary_sampling_rate(transcript, sample_rate):
    """For any transcript with N segments and sample_rate R, the condensed transcript
    should contain approximately N / R segments (±1 for rounding).
    
    This test validates that the video summary generation correctly samples the transcript
    at the specified rate. The sampling strategy is:
    1. Calculate actual_sample_rate = max(1, len(segments) // sample_rate)
    2. Sample segments using segments[::actual_sample_rate]
    3. The result should have approximately len(segments) / actual_sample_rate segments
    
    The ±1 tolerance accounts for:
    - Integer division rounding
    - The max(1, ...) constraint when there are very few segments
    - Edge cases where the last segment might be included or excluded
    
    **Validates: Requirements 2.3**
    """
    from pipeline.scorer import generate_video_summary
    from unittest.mock import patch
    
    # We need to extract the sampled segments from the condensed transcript
    # Since generate_video_summary calls the LLM, we'll mock it and inspect the prompt
    
    config = Config(work_dir="/tmp/test")
    config.video_summary_sample_rate = sample_rate
    
    N = len(transcript.segments)
    
    # Calculate expected sample rate (matching the implementation)
    expected_sample_rate = max(1, N // sample_rate)
    
    # Calculate expected number of sampled segments
    # This is the number of segments we'd get from segments[::expected_sample_rate]
    expected_sampled_count = len(transcript.segments[::expected_sample_rate])
    
    # Mock the LLM call to capture the prompt and return a dummy summary
    with patch('pipeline.scorer._call_llm') as mock_llm:
        mock_llm.return_value = "Test summary"
        
        # Generate the summary (which internally creates the condensed transcript)
        summary = generate_video_summary(config, transcript, video_path="")
        
        # Extract the prompt that was sent to the LLM
        assert mock_llm.call_count == 1, "LLM should be called exactly once"
        prompt = mock_llm.call_args[0][1]  # Second argument to _call_llm is the prompt
        
        # Count the number of segments in the condensed transcript
        # Each segment is formatted as "[Xs] text" on its own line
        # We can count lines that start with "[" and contain "s]"
        import re
        segment_lines = [line for line in prompt.split('\n') if re.match(r'^\[\d+s\]', line)]
        actual_sampled_count = len(segment_lines)
        
        # Verify the sampled count matches our expectation (±1 for rounding)
        # The ±1 tolerance accounts for:
        # 1. Integer division rounding in the sampling calculation
        # 2. The 500-word limit which might truncate the condensed transcript
        # 3. Edge cases in the slicing operation
        assert abs(actual_sampled_count - expected_sampled_count) <= 1, \
            f"Sampled segment count mismatch: expected ~{expected_sampled_count} " \
            f"(from {N} segments with sample_rate={sample_rate}, " \
            f"actual_sample_rate={expected_sample_rate}), " \
            f"but got {actual_sampled_count} segments in condensed transcript"
        
        # Additional validation: verify the sampling rate relationship
        # actual_sampled_count should be approximately N / expected_sample_rate
        if N >= expected_sample_rate:
            expected_from_formula = N // expected_sample_rate
            # Allow ±1 for rounding and edge cases
            assert abs(actual_sampled_count - expected_from_formula) <= 1, \
                f"Sampling rate formula validation failed: " \
                f"N={N}, expected_sample_rate={expected_sample_rate}, " \
                f"expected_from_formula={expected_from_formula}, " \
                f"actual_sampled_count={actual_sampled_count}"


# ---------------------------------------------------------------------------
# Property 16: Prompt Summary Inclusion
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 16: Prompt Summary Inclusion
@given(
    summary=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), whitelist_characters='.,!?-'),
        min_size=10,
        max_size=200
    ),
    window_text=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), whitelist_characters='.,!?-'),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=20, deadline=None)
def test_prompt_summary_inclusion(summary, window_text):
    """For any video summary string and window transcript, the constructed LLM prompt
    should contain the summary text as a prefix.
    
    This test validates Requirement 2.4: THE LLM_Scorer SHALL prepend the video summary
    to every Context_Window prompt sent to the LLM.
    
    The test verifies that:
    1. When a video_summary is provided to _score_window_with_llm, it appears in the prompt
    2. The summary is formatted as "VIDEO CONTEXT: {summary}" in the prompt
    3. The summary appears before the window transcript content
    
    **Validates: Requirements 2.4**
    """
    from pipeline.models import Segment
    from unittest.mock import patch
    
    # Skip empty or whitespace-only inputs
    assume(len(summary.strip()) > 0)
    assume(len(window_text.strip()) > 0)
    
    # Create a minimal config
    config = Config(work_dir="/tmp/test")
    config.llm_enabled = True
    config.llm_endpoint = "http://localhost:11434/api/generate"
    config.llm_model = "llama3"
    config.min_clip_duration = 30.0
    
    # Create a segment with the window text
    segment = Segment(start=0.0, end=5.0, text=window_text)
    all_segments = [segment]
    
    # Mock the LLM call to capture the prompt
    with patch('pipeline.scorer._call_llm') as mock_llm:
        # Return a valid LLM response
        mock_llm.return_value = (
            "SCORE: 5\n"
            "TITLE: Test Clip\n"
            "DESCRIPTION: A test clip for validation.\n"
            "TAGS: #test #shorts"
        )
        
        # Import the function to test
        from pipeline.scorer import _score_window_with_llm
        
        # Call the function with the video summary
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=all_segments,
            all_audio_scores=[0.5],
            all_raw_rms=[0.3],
            global_rms_mean=0.25,
            global_rms_max=1.0,
            video_summary=summary,
        )
        
        # Verify the LLM was called
        assert mock_llm.call_count == 1, "LLM should be called exactly once"
        
        # Extract the prompt that was sent to the LLM
        prompt = mock_llm.call_args[0][1]  # Second argument to _call_llm is the prompt
        
        # Verify the summary is in the prompt
        assert summary in prompt, \
            f"Video summary should be included in the prompt. " \
            f"Summary: '{summary[:50]}...' not found in prompt"
        
        # Verify the summary is formatted with the VIDEO CONTEXT prefix
        expected_context_line = f"VIDEO CONTEXT: {summary}"
        assert expected_context_line in prompt, \
            f"Video summary should be formatted as 'VIDEO CONTEXT: {{summary}}'. " \
            f"Expected: '{expected_context_line[:80]}...' not found in prompt"
        
        # Verify the summary appears before the window transcript
        # The window transcript section starts with "WINDOW TRANSCRIPT:"
        video_context_pos = prompt.find("VIDEO CONTEXT:")
        window_transcript_pos = prompt.find("WINDOW TRANSCRIPT:")
        
        assert video_context_pos >= 0, \
            "Prompt should contain 'VIDEO CONTEXT:' section"
        assert window_transcript_pos >= 0, \
            "Prompt should contain 'WINDOW TRANSCRIPT:' section"
        assert video_context_pos < window_transcript_pos, \
            f"VIDEO CONTEXT should appear before WINDOW TRANSCRIPT in the prompt. " \
            f"VIDEO CONTEXT at position {video_context_pos}, " \
            f"WINDOW TRANSCRIPT at position {window_transcript_pos}"


# Feature: clip-selection-improvements, Property 16: Prompt Summary Inclusion (empty summary case)
@given(
    window_text=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), whitelist_characters='.,!?-'),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=20, deadline=None)
def test_prompt_without_summary(window_text):
    """For any window transcript with no video summary provided (empty string),
    the prompt should NOT contain the VIDEO CONTEXT section.
    
    This test validates that the video summary is optional and the prompt
    construction handles the case where no summary is provided.
    
    **Validates: Requirements 2.4**
    """
    from pipeline.models import Segment
    from unittest.mock import patch
    
    # Skip empty or whitespace-only inputs
    assume(len(window_text.strip()) > 0)
    
    # Create a minimal config
    config = Config(work_dir="/tmp/test")
    config.llm_enabled = True
    config.llm_endpoint = "http://localhost:11434/api/generate"
    config.llm_model = "llama3"
    config.min_clip_duration = 30.0
    
    # Create a segment with the window text
    segment = Segment(start=0.0, end=5.0, text=window_text)
    all_segments = [segment]
    
    # Mock the LLM call to capture the prompt
    with patch('pipeline.scorer._call_llm') as mock_llm:
        # Return a valid LLM response
        mock_llm.return_value = (
            "SCORE: 5\n"
            "TITLE: Test Clip\n"
            "DESCRIPTION: A test clip for validation.\n"
            "TAGS: #test #shorts"
        )
        
        # Import the function to test
        from pipeline.scorer import _score_window_with_llm
        
        # Call the function WITHOUT a video summary (empty string)
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=all_segments,
            all_audio_scores=[0.5],
            all_raw_rms=[0.3],
            global_rms_mean=0.25,
            global_rms_max=1.0,
            video_summary="",  # Empty summary
        )
        
        # Verify the LLM was called
        assert mock_llm.call_count == 1, "LLM should be called exactly once"
        
        # Extract the prompt that was sent to the LLM
        prompt = mock_llm.call_args[0][1]  # Second argument to _call_llm is the prompt
        
        # Verify the VIDEO CONTEXT section is NOT in the prompt when summary is empty
        assert "VIDEO CONTEXT:" not in prompt, \
            "Prompt should NOT contain 'VIDEO CONTEXT:' section when summary is empty"


# ---------------------------------------------------------------------------
# Property 8: Semantic Similarity Symmetry and Bounds
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------

@st.composite
def clip_pair_strategy(draw):
    """Generate a pair of clips with transcript for semantic similarity testing.
    
    Returns a tuple of (clip_a, clip_b, transcript) where both clips reference
    segments in the transcript.
    """
    from pipeline.models import Clip, Transcript, Segment
    
    # Generate 2-10 segments for the transcript
    num_segments = draw(st.integers(min_value=2, max_value=10))
    
    segments = []
    current_time = 0.0
    
    for i in range(num_segments):
        # Each segment is 3-10 seconds long
        duration = draw(st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        start = current_time
        end = current_time + duration
        
        # Generate text for the segment
        text = draw(st.text(
            alphabet='abcdefghijklmnopqrstuvwxyz ',
            min_size=5,
            max_size=50
        ))
        
        segments.append(Segment(start=start, end=end, text=text))
        current_time = end
    
    transcript = Transcript(segments=segments)
    
    # Generate two clips that reference different segments
    # Clip A: references first half of segments
    num_segments_a = draw(st.integers(min_value=1, max_value=max(1, num_segments // 2)))
    segment_indices_a = list(range(num_segments_a))
    
    clip_a = Clip(
        start=segments[0].start,
        end=segments[num_segments_a - 1].end,
        score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        rank=1,
        segment_indices=segment_indices_a
    )
    
    # Clip B: references second half of segments (or overlapping)
    start_idx_b = draw(st.integers(min_value=0, max_value=num_segments - 1))
    num_segments_b = draw(st.integers(min_value=1, max_value=num_segments - start_idx_b))
    segment_indices_b = list(range(start_idx_b, start_idx_b + num_segments_b))
    
    clip_b = Clip(
        start=segments[start_idx_b].start,
        end=segments[start_idx_b + num_segments_b - 1].end,
        score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        rank=2,
        segment_indices=segment_indices_b
    )
    
    return clip_a, clip_b, transcript


# Feature: clip-selection-improvements, Property 8: Semantic Similarity Symmetry and Bounds
@given(data=clip_pair_strategy())
@settings(max_examples=100, deadline=None)
def test_semantic_similarity_symmetry_and_bounds(data):
    """For any two clip transcripts A and B, the semantic similarity should be
    symmetric (sim(A,B) == sim(B,A)) and bounded in the range [0.0, 1.0].
    
    This test validates two critical properties of semantic similarity:
    
    1. **Symmetry**: The similarity between A and B should equal the similarity
       between B and A. This is a fundamental property of distance/similarity
       metrics and ensures consistent behavior regardless of argument order.
    
    2. **Bounds**: The similarity score must be in the range [0.0, 1.0] where:
       - 0.0 indicates completely different/dissimilar content
       - 1.0 indicates identical content
       - Values in between indicate varying degrees of similarity
    
    The test uses the sentence-transformers model (all-MiniLM-L6-v2) to compute
    embeddings and cosine similarity. The implementation clamps the result to
    [0.0, 1.0] to handle edge cases where cosine similarity might be slightly
    negative due to floating-point precision.
    
    **Validates: Requirements 7.1**
    """
    from pipeline.semantic_dedup import compute_semantic_similarity
    
    clip_a, clip_b, transcript = data
    
    # Skip if either clip has no text (empty segment_indices)
    assume(len(clip_a.segment_indices) > 0)
    assume(len(clip_b.segment_indices) > 0)
    
    # Extract text to verify it's non-empty
    text_a = " ".join(transcript.segments[i].text for i in clip_a.segment_indices 
                      if 0 <= i < len(transcript.segments))
    text_b = " ".join(transcript.segments[i].text for i in clip_b.segment_indices 
                      if 0 <= i < len(transcript.segments))
    
    assume(len(text_a.strip()) > 0)
    assume(len(text_b.strip()) > 0)
    
    # Compute similarity in both directions
    similarity_ab = compute_semantic_similarity(clip_a, clip_b, transcript)
    similarity_ba = compute_semantic_similarity(clip_b, clip_a, transcript)
    
    # Property 1: Symmetry - sim(A, B) should equal sim(B, A)
    # Allow small floating-point tolerance (1e-6)
    assert abs(similarity_ab - similarity_ba) < 1e-6, \
        f"Semantic similarity should be symmetric: " \
        f"sim(A, B) = {similarity_ab:.6f}, sim(B, A) = {similarity_ba:.6f}, " \
        f"difference = {abs(similarity_ab - similarity_ba):.6e}"
    
    # Property 2: Bounds - similarity should be in [0.0, 1.0]
    assert 0.0 <= similarity_ab <= 1.0, \
        f"Semantic similarity should be in range [0.0, 1.0], got: {similarity_ab:.6f}"
    
    assert 0.0 <= similarity_ba <= 1.0, \
        f"Semantic similarity should be in range [0.0, 1.0], got: {similarity_ba:.6f}"
    
    # Additional validation: verify the similarity is a valid float
    assert isinstance(similarity_ab, float), \
        f"Similarity should be a float, got: {type(similarity_ab)}"
    
    assert isinstance(similarity_ba, float), \
        f"Similarity should be a float, got: {type(similarity_ba)}"
    
    # Verify no NaN or infinity values
    import math
    assert not math.isnan(similarity_ab), \
        "Similarity should not be NaN"
    
    assert not math.isinf(similarity_ab), \
        "Similarity should not be infinity"
