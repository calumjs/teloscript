#!/usr/bin/env python3
"""
TELOSCRIPT - Purposeful Agent Orchestration System
A goal-directed agent coordination system that accepts MCP configurations + objectives,
orchestrates autonomous agents to achieve goals, and returns intelligent responses.
"""

import uvicorn
from loguru import logger
from teloscript_mcp.src.api import app

def main():
    """Main entry point for TELOSCRIPT"""
    logger.info("Starting TELOSCRIPT - Purposeful Agent Orchestration System...")
    
    # Configure uvicorn server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("TELOSCRIPT API starting on http://0.0.0.0:8000")
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down TELOSCRIPT...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        raise

if __name__ == "__main__":
    main() 