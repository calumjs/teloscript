#!/usr/bin/env python3
"""
TELOSCRIPT MCP Server Entry Point
Standalone script to run TELOSCRIPT as an MCP server
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server import TeloscriptMCPServer


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TELOSCRIPT MCP Server - Expose TELOSCRIPT as an MCP server for recursive agent orchestration"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="URL of the TELOSCRIPT API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="1.0.0"
    )
    
    args = parser.parse_args()
    
    # Create and run the MCP server
    server = TeloscriptMCPServer(teloscript_api_url=args.api_url)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down TELOSCRIPT MCP Server...")
    except Exception as e:
        print(f"Error running MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()