#!/usr/bin/env python3
"""
TELOSCRIPT MCP Server
Exposes TELOSCRIPT functionality as an MCP server for recursive agent orchestration
"""

import asyncio
import json
import sys
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger

import httpx
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, Prompt, TextContent, ImageContent, EmbeddedResource
)
import mcp.types as types

from .models import (
    MCPToolDefinition, AgentRequest, MCPAgentConfig, AgentGoal, 
    MCPServerConfig, AgentStatus, PurposeEndpointRequest
)


class TeloscriptMCPServer:
    """TELOSCRIPT as an MCP Server for recursive agent orchestration"""
    
    def __init__(self, teloscript_api_url: str = "http://localhost:8000"):
        self.api_url = teloscript_api_url
        self.server = Server("teloscript")
        self._setup_handlers()
        logger.info(f"Initialized TELOSCRIPT MCP Server targeting {teloscript_api_url}")
    
    def _setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="spawn_agent",
                    description="Create and execute a new TELOSCRIPT agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string", 
                                "description": "Goal for the agent to accomplish"
                            },
                            "mcp_servers": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "List of MCP server names to use (e.g., ['filesystem', 'brave-search'])"
                            },
                            "max_iterations": {
                                "type": "integer", 
                                "default": 10,
                                "description": "Maximum iterations for goal completion"
                            },
                            "timeout": {
                                "type": "integer", 
                                "default": 300,
                                "description": "Overall timeout in seconds"
                            }
                        },
                        "required": ["goal"]
                    }
                ),
                Tool(
                    name="check_agent_status",
                    description="Check the status of a running agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string", 
                                "description": "ID of the agent to check"
                            }
                        },
                        "required": ["agent_id"]
                    }
                ),
                Tool(
                    name="execute_purpose_endpoint",
                    description="Execute a predefined purpose endpoint",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint_slug": {
                                "type": "string", 
                                "description": "Slug of the purpose endpoint (e.g., 'handle-github-webhook')"
                            },
                            "input_data": {
                                "type": "object", 
                                "description": "Input data for the endpoint"
                            }
                        },
                        "required": ["endpoint_slug", "input_data"]
                    }
                ),
                Tool(
                    name="list_purpose_endpoints",
                    description="List all available purpose endpoints",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_agent_capabilities",
                    description="List available MCP servers and their capabilities",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="create_workflow_template",
                    description="Create a reusable workflow template",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Template name"},
                            "description": {"type": "string", "description": "Template description"},
                            "steps": {
                                "type": "array", 
                                "items": {"type": "object"},
                                "description": "List of workflow steps"
                            },
                            "input_schema": {
                                "type": "object",
                                "description": "JSON schema for template inputs"
                            }
                        },
                        "required": ["name", "steps"]
                    }
                ),
                Tool(
                    name="list_active_agents",
                    description="List all currently active agents",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="cancel_agent",
                    description="Cancel a running agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string", 
                                "description": "ID of the agent to cancel"
                            }
                        },
                        "required": ["agent_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
            """Handle tool calls"""
            try:
                result = await self._handle_tool_call(name, arguments)
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )
                ]
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({
                            "error": str(e),
                            "tool": name,
                            "arguments": arguments
                        }, indent=2)
                    )
                ]
    
    async def _handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        logger.info(f"Handling tool call: {tool_name} with args: {arguments}")
        
        if tool_name == "spawn_agent":
            return await self._spawn_agent(arguments)
        elif tool_name == "check_agent_status":
            return await self._check_agent_status(arguments["agent_id"])
        elif tool_name == "execute_purpose_endpoint":
            return await self._execute_purpose_endpoint(
                arguments["endpoint_slug"], 
                arguments["input_data"]
            )
        elif tool_name == "list_purpose_endpoints":
            return await self._list_purpose_endpoints()
        elif tool_name == "get_agent_capabilities":
            return await self._get_agent_capabilities()
        elif tool_name == "create_workflow_template":
            return await self._create_workflow_template(arguments)
        elif tool_name == "list_active_agents":
            return await self._list_active_agents()
        elif tool_name == "cancel_agent":
            return await self._cancel_agent(arguments["agent_id"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _spawn_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a new TELOSCRIPT agent via API"""
        try:
            goal = args["goal"]
            mcp_servers = args.get("mcp_servers", ["filesystem"])
            max_iterations = args.get("max_iterations", 10)
            timeout = args.get("timeout", 300)
            
            # Prepare the request payload
            payload = {
                "goal": goal,
                "servers": [],
                "max_iterations": max_iterations,
                "timeout": timeout
            }
            
            # Add MCP server configurations
            for server_name in mcp_servers:
                # Use predefined server configs or create basic ones
                server_config = self._get_server_config(server_name)
                payload["servers"].append(server_config)
            
            logger.info(f"Spawning agent with payload: {payload}")
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 10)) as client:
                response = await client.post(
                    f"{self.api_url}/agents",
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
    
    async def _check_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Check the status of a running agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/agents/{agent_id}/status")
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "agent_id": agent_id,
                        "status_data": response.json()
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
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def _execute_purpose_endpoint(self, endpoint_slug: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a predefined purpose endpoint"""
        try:
            payload = {
                "endpoint_slug": endpoint_slug,
                "input_data": input_data
            }
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
                response = await client.post(
                    f"{self.api_url}/purpose/{endpoint_slug}",
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
                "error": str(e),
                "endpoint_slug": endpoint_slug
            }
    
    async def _list_purpose_endpoints(self) -> Dict[str, Any]:
        """List all available purpose endpoints"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/purpose/endpoints")
                
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
    
    async def _get_agent_capabilities(self) -> Dict[str, Any]:
        """List available MCP servers and their capabilities"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/mcp-configs")
                
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
    
    async def _create_workflow_template(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a reusable workflow template"""
        # This would integrate with a workflow template system
        # For now, return a placeholder implementation
        template = {
            "id": str(uuid.uuid4()),
            "name": args["name"],
            "description": args.get("description", ""),
            "steps": args["steps"],
            "input_schema": args.get("input_schema", {}),
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "template": template,
            "message": "Workflow template created (placeholder implementation)"
        }
    
    async def _list_active_agents(self) -> Dict[str, Any]:
        """List all currently active agents"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/agents")
                
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
    
    async def _cancel_agent(self, agent_id: str) -> Dict[str, Any]:
        """Cancel a running agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(f"{self.api_url}/agents/{agent_id}")
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "agent_id": agent_id,
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
                "error": str(e),
                "agent_id": agent_id
            }
    
    def _get_server_config(self, server_name: str) -> Dict[str, Any]:
        """Get MCP server configuration by name"""
        # Common server configurations
        server_configs = {
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
        
        return server_configs.get(server_name, {
            "name": server_name,
            "command": "npx",
            "args": ["-y", f"@modelcontextprotocol/server-{server_name}"],
            "transport": "stdio"
        })
    
    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="teloscript",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point for the MCP server"""
    server = TeloscriptMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())