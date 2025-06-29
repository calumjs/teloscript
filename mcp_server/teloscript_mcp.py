#!/usr/bin/env python3
"""
TELOSCRIPT MCP Server
A proper MCP server implementation following the official MCP specification.
Can be run as a standalone command like NPX packages.
"""

import asyncio
import argparse
import json
import logging
import sys
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

# Use FastMCP from the official SDK for easier implementation
from mcp.server.fastmcp import FastMCP

# HTTP client for API calls
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get API URL from environment or use default
def get_api_url():
    return os.getenv("TELOSCRIPT_API_URL", "http://localhost:8000")

# Create the FastMCP server instance
mcp = FastMCP("teloscript")


@mcp.tool()
async def spawn_agent(
    goal: str,
    mcp_servers: List[str] = ["filesystem"],
    max_iterations: int = 10,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Create and execute a new TELOSCRIPT agent with specified goal and MCP servers.
    
    Args:
        goal: The goal for the agent to accomplish
        mcp_servers: List of MCP server names to use (e.g., ['filesystem', 'brave-search'])
        max_iterations: Maximum iterations for goal completion (1-50)
        timeout: Overall timeout in seconds (30-1800)
    
    Returns:
        Dictionary with agent creation result and details
    """
    try:
        api_url = get_api_url()
        
        # Build server configurations
        servers = []
        for server_name in mcp_servers:
            config = get_server_config(server_name)
            servers.append(config)
        
        payload = {
            "goal": goal,
            "servers": servers,
            "max_iterations": max_iterations,
            "timeout": timeout
        }
        
        logger.info(f"Spawning agent with goal: {goal}")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 10)) as client:
            response = await client.post(
                f"{api_url}/agents",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "agent_id": result.get("agent_id"),
                    "status": result.get("status"),
                    "message": "Agent spawned successfully",
                    "goal": goal,
                    "mcp_servers": mcp_servers,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "message": "Failed to spawn agent"
                }
                
    except Exception as e:
        logger.error(f"Error spawning agent: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Exception occurred while spawning agent"
        }


@mcp.tool()
async def check_agent_status(agent_id: str) -> Dict[str, Any]:
    """
    Check the execution status and progress of a running agent.
    
    Args:
        agent_id: The unique identifier of the agent to check
    
    Returns:
        Dictionary with agent status information
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/agents/{agent_id}/status")
            
            if response.status_code == 200:
                status_data = response.json()
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "status_data": status_data
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": "Agent not found",
                    "agent_id": agent_id
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "agent_id": agent_id
                }
                
    except Exception as e:
        logger.error(f"Error checking agent status: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def execute_purpose_endpoint(endpoint_slug: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a predefined purpose endpoint with input data.
    
    Args:
        endpoint_slug: The slug identifier of the purpose endpoint
        input_data: Input data to pass to the endpoint
    
    Returns:
        Dictionary with endpoint execution result
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
            response = await client.post(
                f"{api_url}/purpose/{endpoint_slug}",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "endpoint_slug": endpoint_slug,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "endpoint_slug": endpoint_slug
                }
                
    except Exception as e:
        logger.error(f"Error executing purpose endpoint: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def list_purpose_endpoints() -> Dict[str, Any]:
    """
    List all available purpose endpoints with their configurations.
    
    Returns:
        Dictionary with list of available purpose endpoints
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/purpose/endpoints")
            
            if response.status_code == 200:
                endpoints = response.json()
                return {
                    "success": True,
                    "endpoints": endpoints
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        logger.error(f"Error listing purpose endpoints: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def get_agent_capabilities() -> Dict[str, Any]:
    """
    Get information about available MCP servers and their capabilities.
    
    Returns:
        Dictionary with MCP server capabilities information
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/mcp-configs")
            
            if response.status_code == 200:
                configs = response.json()
                return {
                    "success": True,
                    "mcp_servers": configs
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        logger.error(f"Error getting agent capabilities: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def list_active_agents() -> Dict[str, Any]:
    """
    List all currently active agents and their status.
    
    Returns:
        Dictionary with list of active agents
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/agents")
            
            if response.status_code == 200:
                agents = response.json()
                return {
                    "success": True,
                    "active_agents": agents
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        logger.error(f"Error listing active agents: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def cancel_agent(agent_id: str, reason: str = "Cancelled by user") -> Dict[str, Any]:
    """
    Cancel a running agent by its ID.
    
    Args:
        agent_id: The unique identifier of the agent to cancel
        reason: Optional reason for cancellation
    
    Returns:
        Dictionary with cancellation result
    """
    try:
        api_url = get_api_url()
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{api_url}/agents/{agent_id}")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "reason": reason,
                    "message": "Agent cancelled successfully"
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": "Agent not found",
                    "agent_id": agent_id
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "agent_id": agent_id
                }
                
    except Exception as e:
        logger.error(f"Error cancelling agent: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def create_workflow_template(
    name: str,
    steps: List[Dict[str, Any]],
    description: str = "",
    input_schema: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a reusable workflow template for complex multi-agent orchestration.
    
    Args:
        name: Name of the workflow template
        steps: List of workflow steps with dependencies
        description: Description of what this workflow does
        input_schema: JSON schema for template input parameters
    
    Returns:
        Dictionary with created workflow template
    """
    try:
        import uuid
        from datetime import datetime
        
        if input_schema is None:
            input_schema = {}
        
        template = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "steps": steps,
            "input_schema": input_schema,
            "created_at": datetime.utcnow().isoformat(),
            "server_info": {
                "type": "teloscript_mcp",
                "version": "1.0.0"
            }
        }
        
        return {
            "success": True,
            "template": template,
            "message": f"Workflow template '{name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating workflow template: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_server_config(server_name: str) -> Dict[str, Any]:
    """Get MCP server configuration by name"""
    configs = {
        "filesystem": {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            "transport": "stdio"
        },
        "brave-search": {
            "name": "brave-search",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "your-api-key-here"},
            "transport": "stdio"
        },
        "github": {
            "name": "github",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"},
            "transport": "stdio"
        },
        "memory": {
            "name": "memory",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "transport": "stdio"
        },
        "puppeteer": {
            "name": "puppeteer",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "transport": "stdio"
        },
        "tavily": {
            "name": "tavily-mcp",
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.2"],
            "env": {"TAVILY_API_KEY": "your-api-key-here"},
            "transport": "stdio"
        }
    }
    
    return configs.get(server_name, {
        "name": server_name,
        "command": "npx",
        "args": ["-y", f"@modelcontextprotocol/server-{server_name}"],
        "transport": "stdio"
    })


def check_api_service_running(api_url: str) -> bool:
    """Check if the TELOSCRIPT API service is already running"""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{api_url}/health")
            return response.status_code == 200
    except:
        return False


def start_api_service():
    """Start the TELOSCRIPT API service in the background"""
    try:
        # Find the main.py file - check common locations
        possible_paths = [
            Path(__file__).parent.parent / "main.py",  # ../main.py from mcp_server/
            Path("main.py"),  # current directory
            Path("../main.py"),  # parent directory
        ]
        
        main_py_path = None
        for path in possible_paths:
            if path.exists():
                main_py_path = path
                break
        
        if not main_py_path:
            logger.error("Could not find main.py to start TELOSCRIPT API service")
            return None
        
        logger.info(f"Starting TELOSCRIPT API service from {main_py_path}")
        
        # Start the service in the background
        process = subprocess.Popen(
            [sys.executable, str(main_py_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=main_py_path.parent
        )
        
        # Wait a bit for the service to start
        time.sleep(3)
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start TELOSCRIPT API service: {e}")
        return None


def main():
    """Main entry point - can be run as a command like NPX packages"""
    parser = argparse.ArgumentParser(
        description="TELOSCRIPT MCP Server - Enable recursive agent orchestration through MCP protocol",
        prog="teloscript-mcp"
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="URL of the TELOSCRIPT API (default: http://localhost:8000, or TELOSCRIPT_API_URL env var)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't automatically start the TELOSCRIPT API service if not running"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="TELOSCRIPT MCP Server 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Set API URL from command line argument or environment
    if args.api_url:
        os.environ["TELOSCRIPT_API_URL"] = args.api_url
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    api_service_process = None
    
    try:
        api_url = get_api_url()
        logger.info("Starting TELOSCRIPT MCP Server...")
        
        # Check if API service is running, start if needed
        if not check_api_service_running(api_url):
            if args.no_auto_start:
                logger.error(f"TELOSCRIPT API service not running at {api_url}")
                logger.error("Start the service manually or remove --no-auto-start flag")
                sys.exit(1)
            else:
                logger.info(f"TELOSCRIPT API service not running at {api_url}")
                logger.info("Auto-starting TELOSCRIPT API service...")
                api_service_process = start_api_service()
                
                if api_service_process:
                    # Wait for service to be ready
                    for i in range(10):  # Wait up to 10 seconds
                        time.sleep(1)
                        if check_api_service_running(api_url):
                            logger.info("TELOSCRIPT API service started successfully")
                            break
                    else:
                        logger.error("TELOSCRIPT API service failed to start within 10 seconds")
                        if api_service_process:
                            api_service_process.terminate()
                        sys.exit(1)
                else:
                    logger.error("Failed to start TELOSCRIPT API service")
                    sys.exit(1)
        else:
            logger.info(f"TELOSCRIPT API service already running at {api_url}")
        
        logger.info(f"Connecting to TELOSCRIPT API at: {api_url}")
        logger.info("Server ready for MCP connections via stdio")
        
        # Run the FastMCP server
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down TELOSCRIPT MCP Server...")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)
    finally:
        # Clean up API service if we started it
        if api_service_process:
            logger.info("Stopping auto-started TELOSCRIPT API service...")
            api_service_process.terminate()
            try:
                api_service_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_service_process.kill()


if __name__ == "__main__":
    main()