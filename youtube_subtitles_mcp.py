#!/usr/bin/env python3
"""
YouTube Subtitles MCP Server
A Model Context Protocol server that provides YouTube subtitle extraction functionality.
"""

import asyncio
import json
import sys
import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs
import aiohttp
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubtitleItem:
    """Represents a single subtitle item with timing information."""
    text: str
    start_time: float
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "duration": self.duration,
            "timestamp": self.format_timestamp()
        }

    def format_timestamp(self) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        hours = int(self.start_time // 3600)
        minutes = int((self.start_time % 3600) // 60)
        seconds = int(self.start_time % 60)
        milliseconds = int((self.start_time % 1) * 1000)
        return "02d"


class YouTubeSubtitleExtractor:
    """Extracts subtitles from YouTube videos."""

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
        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Try parsing as URL and checking query parameters
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'youtu.be']:
                if parsed_url.hostname == 'youtu.be':
                    return parsed_url.path.lstrip('/')
                query_params = parse_qs(parsed_url.query)
                return query_params.get('v', [None])[0]
        except Exception:
            pass

        return None

    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video information including available subtitles."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # YouTube API endpoint for video info
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch video page: {response.status}")

                html = await response.text()

                # Extract subtitle information from the page
                # This is a simplified version - in production you'd need more robust parsing
                subtitle_info = self._extract_subtitle_info(html)

                return {
                    "video_id": video_id,
                    "subtitles": subtitle_info
                }

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    def _extract_subtitle_info(self, html: str) -> Dict[str, Any]:
        """Extract subtitle information from YouTube page HTML."""
        # Look for subtitle data in the page
        # This is a basic implementation - YouTube's structure can change

        subtitles = {}

        # Try to find subtitle tracks in the page source
        # This is a simplified approach - in reality you'd need more sophisticated parsing
        subtitle_patterns = [
            r'"captionTracks":\s*\[(.*?)\]',
            r'"subtitles":\s*\{(.*?)\}',
        ]

        for pattern in subtitle_patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            if matches:
                try:
                    # Basic parsing of JSON-like structures
                    for match in matches:
                        # This would need proper JSON parsing in a real implementation
                        if 'en' in match.lower() or 'english' in match.lower():
                            subtitles['en'] = {"available": True}
                        if 'zh' in match.lower() or 'chinese' in match.lower():
                            subtitles['zh'] = {"available": True}
                except Exception:
                    continue

        return subtitles

    async def get_subtitles(self, video_id: str, lang: str = 'en') -> List[SubtitleItem]:
        """Get subtitles for a specific video and language."""
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            # First get video info to check subtitle availability
            video_info = await self.get_video_info(video_id)

            # Try multiple subtitle extraction approaches
            subtitle_sources = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&kind=asr",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&kind=captions",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&tlang={lang}",
                f"https://www.youtube.com/api/timedtext?v={video_id}",
                f"https://www.youtube.com/api/timedtext?v={video_id}&kind=asr",
                f"https://www.youtube.com/api/timedtext?v={video_id}&kind=captions",
            ]

            for subtitle_url in subtitle_sources:
                try:
                    logger.info(f"Trying subtitle URL: {subtitle_url}")
                    async with self.session.get(subtitle_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            xml_content = await response.text()
                            if xml_content and '<transcript>' in xml_content or '<text' in xml_content:
                                # Parse the XML subtitle format
                                subtitles = self._parse_subtitle_xml(xml_content)
                                if subtitles:
                                    logger.info(f"Successfully extracted {len(subtitles)} subtitles from {subtitle_url}")
                                    return subtitles
                        else:
                            logger.debug(f"Subtitle URL {subtitle_url} returned status {response.status}")
                except Exception as e:
                    logger.debug(f"Failed to fetch from {subtitle_url}: {e}")
                    continue

            # If no subtitles found with any method, raise exception
            raise Exception(f"No subtitles found for video {video_id} with language {lang}")

        except Exception as e:
            logger.error(f"Error getting subtitles: {e}")
            raise

    def _parse_subtitle_xml(self, xml_content: str) -> List[SubtitleItem]:
        """Parse YouTube's subtitle XML format."""
        subtitles = []

        try:
            # Handle different YouTube subtitle XML formats

            # Format 1: Standard timed text format
            # <text start="0.5" dur="2.5">Hello world</text>
            patterns = [
                r'<text[^>]*start="([^"]*)"[^>]*dur="([^"]*)"[^>]*>(.*?)</text>',
                r'<text[^>]*start="([^"]*)"[^>]*duration="([^"]*)"[^>]*>(.*?)</text>',
                r'<s[^>]*start="([^"]*)"[^>]*dur="([^"]*)"[^>]*>(.*?)</s>',
                r'<p[^>]*start="([^"]*)"[^>]*dur="([^"]*)"[^>]*>(.*?)</p>',
            ]

            matches = []
            for pattern in patterns:
                matches = re.findall(pattern, xml_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    logger.info(f"Found {len(matches)} subtitles using pattern: {pattern}")
                    break

            if not matches:
                # Try alternative format - look for any text elements
                alt_pattern = r'<text[^>]*>(.*?)</text>'
                alt_matches = re.findall(alt_pattern, xml_content, re.DOTALL | re.IGNORECASE)
                if alt_matches:
                    logger.info(f"Found {len(alt_matches)} subtitles using alternative pattern")
                    # For alternative format, we'll need to estimate timing
                    for i, alt_text in enumerate(alt_matches):
                        try:
                            # Clean up the text content
                            alt_text = re.sub(r'<[^>]+>', '', alt_text)
                            alt_text = alt_text.replace('&', '&')
                            alt_text = alt_text.replace('<', '<')
                            alt_text = alt_text.replace('>', '>')
                            alt_text = alt_text.replace('"', '"')
                            alt_text = alt_text.replace(''', "'")

                            if alt_text.strip():
                                # Estimate timing based on position
                                estimated_start_time = i * 3.0  # 3 seconds per subtitle estimate
                                subtitle_item = SubtitleItem(
                                    text=alt_text.strip(),
                                    start_time=estimated_start_time,
                                    duration=3.0
                                )
                                subtitles.append(subtitle_item)
                        except Exception as e:
                            logger.warning(f"Failed to parse alternative subtitle {i}: {e}")
                            continue
                else:
                    logger.warning("No subtitle patterns found in XML content")
                return subtitles

            # Parse matches with timing information
            for start, duration, text in matches:
                try:
                    start_time = float(start)
                    dur = float(duration)

                    # Clean up the text content
                    text = re.sub(r'<[^>]+>', '', text)  # Remove any remaining HTML tags
                    text = text.replace('&', '&')
                    text = text.replace('<', '<')
                    text = text.replace('>', '>')
                    text = text.replace('"', '"')
                    text = text.replace(''', "'")

                    if text.strip():  # Only add non-empty text
                        subtitle_item = SubtitleItem(
                            text=text.strip(),
                            start_time=start_time,
                            duration=dur
                        )
                        subtitles.append(subtitle_item)

                except ValueError as e:
                    logger.warning(f"Failed to parse subtitle timing: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing subtitle: {e}")
                    continue

            logger.info(f"Successfully parsed {len(subtitles)} subtitles")
            return subtitles

        except Exception as e:
            logger.error(f"Error parsing subtitle XML: {e}")
            return []


class YouTubeSubtitlesMCPServer:
    """MCP Server for YouTube subtitle extraction."""

    def __init__(self):
        self.extractor: Optional[YouTubeSubtitleExtractor] = None

    async def initialize(self):
        """Initialize the server and extractor."""
        self.extractor = YouTubeSubtitleExtractor()

    async def shutdown(self):
        """Shutdown the server."""
        pass

    async def handle_get_subtitles(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the get_subtitles tool call."""
        try:
            url = arguments.get('url', '')
            language = arguments.get('language', 'en')

            if not url:
                return {
                    "error": "URL is required",
                    "subtitles": []
                }

            if not self.extractor:
                return {
                    "error": "Extractor not initialized",
                    "subtitles": []
                }

            # Extract video ID from URL
            video_id = self.extractor.extract_video_id(url)
            if not video_id:
                return {
                    "error": "Invalid YouTube URL or unable to extract video ID",
                    "subtitles": []
                }

            # Get subtitles
            async with self.extractor as ext:
                subtitles = await ext.get_subtitles(video_id, language)

                return {
                    "video_id": video_id,
                    "language": language,
                    "subtitle_count": len(subtitles),
                    "subtitles": [sub.to_dict() for sub in subtitles]
                }

        except Exception as e:
            logger.error(f"Error in get_subtitles: {e}")
            return {
                "error": str(e),
                "subtitles": []
            }


async def main():
    """Main entry point for the MCP server."""
    server = YouTubeSubtitlesMCPServer()
    await server.initialize()

    try:
        # Read messages from stdin
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                message = json.loads(line.strip())
                logger.info(f"Received message: {message}")

                # Handle different message types
                if message.get('method') == 'tools/call':
                    params = message.get('params', {})
                    tool_name = params.get('name', '')

                    if tool_name == 'get_subtitles':
                        result = await server.handle_get_subtitles(params.get('arguments', {}))

                        # Send response
                        response = {
                            'jsonrpc': '2.0',
                            'id': message.get('id'),
                            'result': result
                        }
                        print(json.dumps(response), flush=True)
                    else:
                        # Unknown tool
                        response = {
                            'jsonrpc': '2.0',
                            'id': message.get('id'),
                            'error': {
                                'code': -32601,
                                'message': f'Unknown tool: {tool_name}'
                            }
                        }
                        print(json.dumps(response), flush=True)

                elif message.get('method') == 'initialize':
                    # Handle initialization
                    response = {
                        'jsonrpc': '2.0',
                        'id': message.get('id'),
                        'result': {
                            'protocolVersion': '2024-11-05',
                            'capabilities': {
                                'tools': {}
                            },
                            'serverInfo': {
                                'name': 'youtube-subtitles',
                                'version': '1.0.0'
                            }
                        }
                    }
                    print(json.dumps(response), flush=True)

                elif message.get('method') == 'tools/list':
                    # List available tools
                    response = {
                        'jsonrpc': '2.0',
                        'id': message.get('id'),
                        'result': {
                            'tools': [
                                {
                                    'name': 'get_subtitles',
                                    'description': 'Extract subtitles from YouTube videos',
                                    'inputSchema': {
                                        'type': 'object',
                                        'properties': {
                                            'url': {
                                                'type': 'string',
                                                'description': 'YouTube video URL'
                                            },
                                            'language': {
                                                'type': 'string',
                                                'description': 'Subtitle language (default: en)',
                                                'default': 'en'
                                            }
                                        },
                                        'required': ['url']
                                    }
                                }
                            ]
                        }
                    }
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                logger.error("Failed to parse JSON message")
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())