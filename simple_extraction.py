#!/usr/bin/env python3
"""
Simple YouTube subtitle extraction for the given URL
"""

import asyncio
import re
import aiohttp
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SubtitleItem:
    """Represents a single subtitle item with timing information."""
    text: str
    start_time: float
    duration: float = 0.0

    def format_timestamp(self) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        hours = int(self.start_time // 3600)
        minutes = int((self.start_time % 3600) // 60)
        seconds = int(self.start_time % 60)
        milliseconds = int((self.start_time % 1) * 1000)
        return "02d"


class SimpleYouTubeExtractor:
    """Simple YouTube subtitle extractor."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Try parsing as URL and checking query parameters
        try:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'youtu.be']:
                if parsed_url.hostname == 'youtu.be':
                    return parsed_url.path.lstrip('/')
                query_params = parse_qs(parsed_url.query)
                return query_params.get('v', [None])[0]
        except Exception:
            pass

        return None

    async def get_subtitles(self, video_id: str) -> List[SubtitleItem]:
        """Get subtitles for a video."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # Try different subtitle sources
        subtitle_sources = [
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&kind=asr",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&kind=captions",
            f"https://www.youtube.com/api/timedtext?v={video_id}",
        ]

        for subtitle_url in subtitle_sources:
            try:
                async with self.session.get(subtitle_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        if xml_content and ('<text' in xml_content or '<transcript>' in xml_content):
                            subtitles = self._parse_subtitle_xml(xml_content)
                            if subtitles:
                                return subtitles
            except Exception:
                continue

        return []

    def _parse_subtitle_xml(self, xml_content: str) -> List[SubtitleItem]:
        """Parse YouTube's subtitle XML format."""
        subtitles = []

        # Try different XML patterns
        patterns = [
            r'<text[^>]*start="([^"]*)"[^>]*dur="([^"]*)"[^>]*>(.*?)</text>',
            r'<text[^>]*start="([^"]*)"[^>]*duration="([^"]*)"[^>]*>(.*?)</text>',
            r'<s[^>]*start="([^"]*)"[^>]*dur="([^"]*)"[^>]*>(.*?)</s>',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, xml_content, re.DOTALL | re.IGNORECASE)
            if matches:
                break

        for start, duration, text in matches:
            try:
                start_time = float(start)
                dur = float(duration)

                # Clean up text
                text = re.sub(r'<[^>]+>', '', text)
                text = text.replace('&', '&')
                text = text.replace('<', '<')
                text = text.replace('>', '>')
                text = text.replace('"', '"')
                text = text.replace(''', "'")

                if text.strip():
                    subtitle_item = SubtitleItem(
                        text=text.strip(),
                        start_time=start_time,
                        duration=dur
                    )
                    subtitles.append(subtitle_item)

            except (ValueError, Exception):
                continue

        return subtitles


async def main():
    """Main function."""
    url = "https://youtu.be/q4pQz5qu9QE?si=EyPP8QDNAhAQdBYp"

    print(f"🔍 Extracting subtitles from: {url}")

    async with SimpleYouTubeExtractor() as extractor:
        # Extract video ID
        video_id = extractor.extract_video_id(url)
        if not video_id:
            print("❌ Failed to extract video ID")
            return

        print(f"✅ Video ID: {video_id}")

        # Get subtitles
        subtitles = await extractor.get_subtitles(video_id)

        if subtitles:
            print("
📝 SUBTITLES FOUND!"            print("=" * 50)
            for i, sub in enumerate(subtitles):
                print(f"[{sub.format_timestamp()}] {sub.text}")
            print("=" * 50)
            print(f"\n✅ Total: {len(subtitles)} subtitles")
        else:
            print("❌ No subtitles found for this video")
            print("This could mean:")
            print("  - The video has no auto-generated subtitles")
            print("  - Only manual captions are available")
            print("  - The video content doesn't trigger subtitle generation")


if __name__ == "__main__":
    asyncio.run(main())