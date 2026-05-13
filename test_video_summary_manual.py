#!/usr/bin/env python3
"""Manual integration test for generate_video_summary().

This script demonstrates the video summary generation feature.
Run with: python3 test_video_summary_manual.py
"""

from config import Config
from pipeline.models import Segment, Transcript
from pipeline.scorer import generate_video_summary

def main():
    # Create a realistic gaming transcript
    segments = []
    
    # Simulate a 5-minute gaming video with various moments
    gaming_phrases = [
        "Oh my god, did you see that?",
        "That was insane!",
        "Let me try this again.",
        "Watch this, watch this!",
        "No way, I can't believe it!",
        "This is going to be epic.",
        "Wait, what just happened?",
        "I'm going to clutch this.",
        "Let's go! That was perfect!",
        "Are you kidding me right now?",
    ]
    
    for i in range(100):
        start = i * 3.0
        end = start + 3.0
        text = gaming_phrases[i % len(gaming_phrases)]
        segments.append(Segment(start=start, end=end, text=text))
    
    transcript = Transcript(segments=segments)
    
    # Create config
    config = Config(work_dir="/tmp/test")
    config.llm_enabled = True
    config.llm_endpoint = "http://localhost:11434/api/generate"
    config.llm_model = "llama3"
    
    print("=" * 80)
    print("VIDEO SUMMARY GENERATION TEST")
    print("=" * 80)
    print(f"\nTranscript: {len(segments)} segments, {segments[-1].end / 60.0:.1f} minutes")
    print(f"Sample rate: {len(segments) // 20}")
    print(f"Expected sampled segments: ~20")
    print("\nGenerating video summary...")
    print("-" * 80)
    
    try:
        summary = generate_video_summary(config, transcript, video_path="/test/gaming_video.mp4")
        
        print("\nGENERATED SUMMARY:")
        print("-" * 80)
        print(summary)
        print("-" * 80)
        
        print("\n✓ Video summary generated successfully!")
        print(f"✓ Summary length: {len(summary)} characters")
        print(f"✓ Summary cached: {'/test/gaming_video.mp4' in generate_video_summary.__globals__['_video_summary_cache']}")
        
        # Test cache retrieval
        print("\nTesting cache retrieval...")
        summary2 = generate_video_summary(config, transcript, video_path="/test/gaming_video.mp4")
        print(f"✓ Cache hit: {summary == summary2}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: This test requires a running Ollama instance with llama3 model.")
        print("Start Ollama with: ollama serve")
        print("Pull model with: ollama pull llama3")
        return 1
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    exit(main())
