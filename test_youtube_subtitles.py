#!/usr/bin/env python3
"""
Test script for YouTube Subtitles MCP Server
"""

import asyncio
import json
import sys
import subprocess
from pathlib import Path


async def test_youtube_subtitles():
    """Test the YouTube subtitles MCP server with the provided URL."""

    # The URL provided by the user
    test_url = "https://youtu.be/q4pQz5qu9QE?si=EyPP8QDNAhAQdBYp"

    print("Testing YouTube Subtitles MCP Server...")
    print(f"Test URL: {test_url}")
    print("-" * 50)

    try:
        # Start the MCP server as a subprocess
        server_process = subprocess.Popen(
            [sys.executable, "youtube_subtitles_mcp.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent
        )

        if server_process.stdin is None or server_process.stdout is None:
            print("❌ Failed to create subprocess pipes")
            return

        # Initialize the server
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }

        server_process.stdin.write(json.dumps(init_message) + "\n")
        server_process.stdin.flush()

        # Wait a bit for initialization
        await asyncio.sleep(0.5)

        # Call the get_subtitles tool
        tool_call_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_subtitles",
                "arguments": {
                    "url": test_url,
                    "language": "en"
                }
            }
        }

        server_process.stdin.write(json.dumps(tool_call_message) + "\n")
        server_process.stdin.flush()

        # Read the response
        response = server_process.stdout.readline()
        if response:
            try:
                result = json.loads(response.strip())
                print("Server Response:")
                print(json.dumps(result, indent=2))

                if 'result' in result:
                    result_data = result['result']
                    if 'error' in result_data:
                        print(f"\n❌ Error: {result_data['error']}")
                    else:
                        print("\n✅ Success!")
                        print(f"Video ID: {result_data.get('video_id', 'N/A')}")
                        print(f"Language: {result_data.get('language', 'N/A')}")
                        print(f"Subtitle count: {result_data.get('subtitle_count', 0)}")

                        if result_data.get('subtitles'):
                            print("\n📝 First few subtitles:")
                            for i, sub in enumerate(result_data['subtitles'][:3]):
                                print(f"  {i+1}. [{sub['timestamp']}] {sub['text'][:100]}...")
                elif 'error' in result:
                    print(f"❌ Server Error: {result['error']}")
                else:
                    print("❌ Unexpected response format")

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response: {response}")
            except Exception as e:
                print(f"❌ Error processing response: {e}")
        else:
            print("❌ No response from server")

        # Clean up
        server_process.terminate()
        server_process.wait()

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


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
    asyncio.run(test_youtube_subtitles())


if __name__ == "__main__":
    main()