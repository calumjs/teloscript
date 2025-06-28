#!/usr/bin/env python3
"""
Setup script for TELOSCRIPT MCP Server
"""

from setuptools import setup, find_packages

setup(
    name="teloscript-mcp",
    version="1.0.0",
    description="TELOSCRIPT MCP Server - Recursive agent orchestration through MCP protocol",
    author="TELOSCRIPT Team",
    packages=find_packages(),
    
    # Entry points - this makes it installable as a command
    entry_points={
        "console_scripts": [
            "teloscript-mcp=src.teloscript_mcp:main",
            "teloscript-mcp-server=src.teloscript_mcp:main",
        ],
    },
    
    # Dependencies
    install_requires=[
        "mcp>=1.1.0",
        "httpx>=0.27.0",
        "typing-extensions>=4.8.0",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
)