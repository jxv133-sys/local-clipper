"""Phrase detection module for multi-word keyword matching.

This module implements phrase detection for the text scoring pipeline,
matching multi-word phrases like "oh my god" as atomic units with word
boundaries enforced.

**Validates: Requirements 4.1, 4.2, 4.4, 4.6**
"""

import re
from typing import List, Tuple


def detect_phrases(text: str, phrases: List[str]) -> List[Tuple[str, int, int]]:
    """Find all phrase matches in text with positions.
    
    Matches phrases case-insensitively with word boundaries enforced.
    For example, "oh my god" will match "Oh My God!" but not "ohmygod".
    
    Args:
        text: Input text (e.g., segment.text)
        phrases: List of multi-word phrases (e.g., ["oh my god", "no way"])
    
    Returns:
        List of (phrase, start_pos, end_pos) tuples, sorted by start_pos.
        Overlapping phrases are all included in the results.
    
    Algorithm:
        1. Normalize text: lowercase for matching, preserve original positions
        2. For each phrase:
           a. Build regex: r'\b' + re.escape(phrase) + r'\b'
           b. Find all matches with positions using re.finditer()
        3. Sort results by start_pos
        4. Return all matches (including overlaps)
    
    Examples:
        >>> detect_phrases("Oh my god, no way!", ["oh my god", "no way"])
        [('oh my god', 0, 9), ('no way', 11, 17)]
        
        >>> detect_phrases("ohmygod", ["oh my god"])
        []  # No match due to word boundary requirement
        
        >>> detect_phrases("Oh my, oh my god!", ["oh my", "oh my god"])
        [('oh my', 0, 5), ('oh my', 7, 12), ('oh my god', 7, 17)]  # Overlapping matches
    """
    if not text or not phrases:
        return []
    
    matches: List[Tuple[str, int, int]] = []
    
    # Process each phrase
    for phrase in phrases:
        if not phrase:
            continue
        
        # Build regex pattern with word boundaries
        # re.escape() handles special regex characters in the phrase
        # re.IGNORECASE flag for case-insensitive matching
        pattern = r'\b' + re.escape(phrase) + r'\b'
        
        # Find all matches in the text
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_pos = match.start()
            end_pos = match.end()
            # Store the original phrase (not the matched text) for consistency
            matches.append((phrase, start_pos, end_pos))
    
    # Sort by start position for consistent ordering
    matches.sort(key=lambda x: x[1])
    
    return matches
