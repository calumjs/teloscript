#!/usr/bin/env python3
"""
Setup script for TELOSCRIPT MCP Server
Makes it installable as a command-line tool like NPX packages
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "MCP_SERVER_README.md").read_text()

setup(
    name="teloscript-mcp",
    version="1.0.0",
    description="TELOSCRIPT MCP Server - Enable recursive agent orchestration through MCP protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TELOSCRIPT Team",
    author_email="support@teloscript.dev",
    url="https://github.com/calumjs/teloscript",
    packages=find_packages(),
    py_modules=["teloscript_mcp"],
    
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
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Keywords
    keywords="mcp, ai, agents, orchestration, recursive, automation, teloscript",
    
    # Include package data
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/calumjs/teloscript/issues",
        "Source": "https://github.com/calumjs/teloscript",
        "Documentation": "https://github.com/calumjs/teloscript/blob/main/MCP_SERVER_README.md",
    },
)