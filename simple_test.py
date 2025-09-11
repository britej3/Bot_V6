#!/usr/bin/env python3
"""
Simple test for YouTube subtitle extraction without MCP protocol
"""

import asyncio
import sys
sys.path.append('.')

from youtube_subtitles_mcp import YouTubeSubtitleExtractor


async def test_subtitle_extraction():
    """Test subtitle extraction directly."""
    url = "https://youtu.be/q4pQz5qu9QE?si=EyPP8QDNAhAQdBYp"

    async with YouTubeSubtitleExtractor() as extractor:
        print(f"Testing URL: {url}")

        # Extract video ID
        video_id = extractor.extract_video_id(url)
        if not video_id:
            print("❌ Failed to extract video ID from URL")
            return

        print(f"✅ Extracted video ID: {video_id}")

        # Try to get video info
        try:
            video_info = await extractor.get_video_info(video_id)
            print(f"✅ Video info retrieved: {video_info}")
        except Exception as e:
            print(f"❌ Failed to get video info: {e}")
            return

        # Try different subtitle extraction approaches
        languages = ['en', '', 'asr', 'auto']
        found_subtitles = False

        for lang in languages:
            try:
                print(f"\n🔍 Trying language: '{lang}'")
                subtitles = await extractor.get_subtitles(video_id, lang)
                print(f"✅ Retrieved {len(subtitles)} subtitles for language '{lang}'")

                if subtitles:
                    found_subtitles = True
                    print("\n📝 SUBTITLES FOUND!")
                    print("=" * 50)
                    for i, sub in enumerate(subtitles):
                        print(f"[{sub.format_timestamp()}] {sub.text}")
                    print("=" * 50)
                    break
                else:
                    print(f"⚠️ No subtitles for language '{lang}'")

            except Exception as e:
                print(f"❌ Failed to get subtitles for language '{lang}': {e}")

        if not found_subtitles:
            print("\n❌ No subtitles found with any method")
            print("This could mean:")
            print("  - The video has no auto-generated subtitles")
            print("  - Only manual captions are available")
            print("  - YouTube's API restrictions")
            print("  - The video might be too recent or processing")

        # Try alternative method - check for manual captions
        print("\n🔍 Checking for manual captions...")
        try:
            # This would require additional YouTube API integration
            print("Manual caption detection would require YouTube API v3")
        except Exception as e:
            print(f"❌ Manual caption check failed: {e}")


def main():
    """Main entry point."""
    print("YouTube Subtitles MCP Server Test")
    print("=" * 40)

    # Check if required dependencies are installed
    try:
        import aiohttp
        print("✅ aiohttp is installed")
    except ImportError:
        print("❌ aiohttp is not installed. Please run: pip install -r requirements-youtube-mcp.txt")
        return

    # Run the async test
    asyncio.run(test_subtitle_extraction())


if __name__ == "__main__":
    main()