#!/usr/bin/env python3
"""
Simple launcher for TELOSCRIPT MCP Server
This avoids import-time issues by delaying imports until runtime
"""

def main():
    """Entry point that imports and runs the actual main function"""
    try:
        from .teloscript_mcp import main as teloscript_main
        teloscript_main()
    except ImportError:
        # Fallback for different import contexts
        from teloscript_mcp import main as teloscript_main
        teloscript_main()

if __name__ == "__main__":
    main() 