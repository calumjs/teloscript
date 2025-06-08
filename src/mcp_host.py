import asyncio
import json
import logging
import subprocess
import platform
from typing import Dict, Any, Optional, List
import httpx
import uuid
import time
import sys
import os

# Fix imports
try:
    from .models import (
        MCPServerConfig, MCPServerConnection, MCPToolDefinition,
        MCPResourceDefinition, MCPPromptDefinition, AgentGoal,
        ToolInvocation, ResourceRequest, PromptInvocation,
        StatusUpdate, AgentStatus
    )
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from models import (
        MCPServerConfig, MCPServerConnection, MCPToolDefinition,
        MCPResourceDefinition, MCPPromptDefinition, AgentGoal,
        ToolInvocation, ResourceRequest, PromptInvocation,
        StatusUpdate, AgentStatus
    )

logger = logging.getLogger(__name__)

class MCPProtocolError(Exception):
    """MCP protocol related errors"""
    pass

class MCPTransport:
    """Base class for MCP transports"""
    
    async def connect(self) -> None:
        """Connect to the MCP server"""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        raise NotImplementedError
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the MCP server"""
        raise NotImplementedError
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message from the MCP server"""
        raise NotImplementedError

class StdioTransport(MCPTransport):
    """stdio transport for MCP servers"""
    
    def __init__(self, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        
    async def connect(self) -> None:
        """Start the MCP server process"""
        try:
            logger.info(f"🚀 Starting MCP server: {self.command} {' '.join(self.args)}")
            logger.info(f"📋 Environment variables provided: {list(self.env.keys()) if self.env else 'None'}")
            
            # Prepare environment for subprocess
            process_env = os.environ.copy()  # Start with current environment
            if self.env:
                logger.info(f"🔧 Merging environment variables: {self.env}")
                process_env.update(self.env)
                # Log which env vars we're setting (without values for security)
                for key in self.env.keys():
                    logger.info(f"  • {key}: {'***' if 'token' in key.lower() or 'key' in key.lower() else self.env[key]}")
            
            # Determine the actual command to run
            cmd_args = self._prepare_command()
            
            logger.info(f"🏃 Executing: {' '.join(cmd_args)}")
            
            # Use asyncio subprocess for proper async handling
            self.process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env  # This is the key fix - passing environment variables!
            )
            
            logger.info(f"✅ Started MCP server successfully")
            logger.info(f"   • Command: {self.command}")
            logger.info(f"   • Arguments: {self.args}")
            logger.info(f"   • PID: {self.process.pid}")
            logger.info(f"   • Environment vars: {list(self.env.keys()) if self.env else 'None'}")
            
            # Give the process a moment to start up
            await asyncio.sleep(0.5)
            
            # Check if process is still running
            if self.process.returncode is not None:
                # Process already exited, capture stderr
                try:
                    stderr_output = await asyncio.wait_for(
                        self.process.stderr.read(), 
                        timeout=2.0
                    )
                    if stderr_output:
                        error_msg = stderr_output.decode().strip()
                        logger.error(f"❌ MCP server exited immediately with code {self.process.returncode}")
                        logger.error(f"❌ stderr: {error_msg}")
                        raise MCPProtocolError(f"MCP server failed to start - exited with code {self.process.returncode}: {error_msg}")
                    else:
                        logger.error(f"❌ MCP server exited immediately with code {self.process.returncode} (no stderr)")
                        raise MCPProtocolError(f"MCP server failed to start - exited with code {self.process.returncode}")
                except asyncio.TimeoutError:
                    logger.error(f"❌ MCP server exited immediately with code {self.process.returncode}")
                    raise MCPProtocolError(f"MCP server failed to start - exited with code {self.process.returncode}")
            
        except Exception as e:
            logger.error(f"💥 Failed to start MCP server: {e}")
            logger.error(f"   • Command: {self.command}")
            logger.error(f"   • Arguments: {self.args}")
            logger.error(f"   • Environment: {list(self.env.keys()) if self.env else 'None'}")
            raise MCPProtocolError(f"Failed to start MCP server: {e}")
    
    def _prepare_command(self) -> List[str]:
        """Prepare the command arguments, handling globally installed packages"""
        # Check if this is an MCP package that might be globally installed
        if self.command.startswith("@modelcontextprotocol/"):
            logger.info(f"🔍 Detected MCP package: {self.command}")
            
            # Check if package is globally installed by trying to find it
            try:
                import subprocess
                # Try to find the global package
                result = subprocess.run(
                    ["npm", "list", "-g", "--depth=0", self.command],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ Found globally installed package: {self.command}")
                    # Use npx to run the globally installed package
                    cmd_args = ["npx", self.command] + self.args
                    logger.info(f"💡 Using globally installed package via npx")
                    return self._platform_specific_command(cmd_args)
                else:
                    logger.info(f"⚠️ Package not globally installed, falling back to npx -y")
                    # Fall back to npx -y for download and run
                    cmd_args = ["npx", "-y", self.command] + self.args
                    logger.info(f"💡 Using npx -y fallback for: {self.command}")
                    return self._platform_specific_command(cmd_args)
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not check global package installation: {e}")
                # Fall back to npx -y
                cmd_args = ["npx", "-y", self.command] + self.args
                logger.info(f"💡 Using npx -y fallback due to check error")
                return self._platform_specific_command(cmd_args)
        
        # For non-MCP packages or other commands, use as-is
        cmd_args = [self.command] + self.args
        return self._platform_specific_command(cmd_args)
    
    def _platform_specific_command(self, cmd_args: List[str]) -> List[str]:
        """Apply platform-specific command handling"""
        if platform.system() == "Windows":
            # Check if command is a PowerShell script (like npx)
            if cmd_args[0] == "npx":
                # Use the cmd version of npx for better subprocess compatibility
                npx_cmd_path = r"C:\Program Files\nodejs\npx.cmd"
                if not os.path.exists(npx_cmd_path):
                    # Fallback to system PATH
                    logger.info(f"💡 Using system PATH npx (cmd version not found)")
                    return cmd_args
                else:
                    logger.info(f"💡 Using direct npx.cmd path: {npx_cmd_path}")
                    return [npx_cmd_path] + cmd_args[1:]
            elif cmd_args[0] in ["npm", "yarn"] or cmd_args[0].endswith(".ps1"):
                # Use cmd.exe to run other commands
                logger.info(f"💡 Using cmd.exe wrapper for: {cmd_args[0]}")
                return ["cmd", "/c"] + cmd_args
            else:
                # Regular executable
                logger.info(f"💡 Using direct executable: {cmd_args[0]}")
                return cmd_args
        else:
            # Unix-like systems
            logger.info(f"💡 Unix command: {cmd_args}")
            return cmd_args
    
    async def disconnect(self) -> None:
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON-RPC message via stdin"""
        if not self.process or not self.process.stdin:
            raise MCPProtocolError("MCP server not connected")
        
        try:
            json_str = json.dumps(message)
            logger.debug(f"📤 Sending message to MCP server: {json_str}")
            self.process.stdin.write((json_str + '\n').encode())
            await self.process.stdin.drain()
            logger.debug(f"✅ Message sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            logger.error(f"   • Message was: {message}")
            raise MCPProtocolError(f"Failed to send message: {e}")
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive JSON-RPC message via stdout"""
        if not self.process or not self.process.stdout:
            raise MCPProtocolError("MCP server not connected")
        
        logger.debug("📥 Starting to receive message from MCP server...")
        
        # Some MCP servers output informational text before JSON-RPC
        # Keep reading lines until we get valid JSON
        max_attempts = 20  # Increased from 10
        attempts = 0
        
        while attempts < max_attempts:
            try:
                logger.debug(f"🔄 Attempt {attempts + 1}/{max_attempts} to read line...")
                # Use asyncio readline with timeout
                line_bytes = await asyncio.wait_for(
                    self.process.stdout.readline(), 
                    timeout=15.0  # Increased from 10.0
                )
                
                logger.debug(f"📋 Read {len(line_bytes)} bytes from stdout")
                
                if not line_bytes:
                    logger.warning("⚠️ No data received from MCP server stdout")
                    
                    # Check if process has stderr output before failing
                    try:
                        logger.debug("🔍 Checking stderr for error messages...")
                        stderr_data = await asyncio.wait_for(
                            self.process.stderr.read(), 
                            timeout=1.0
                        )
                        if stderr_data:
                            stderr_msg = stderr_data.decode().strip()
                            logger.error(f"❌ MCP server stderr: {stderr_msg}")
                            raise MCPProtocolError(f"MCP server error: {stderr_msg}")
                    except asyncio.TimeoutError:
                        logger.debug("ℹ️ No stderr data available")
                        pass
                    
                    # Check if process is still running
                    if self.process.returncode is not None:
                        logger.error(f"💀 MCP server process exited with code: {self.process.returncode}")
                        raise MCPProtocolError(f"MCP server exited with code: {self.process.returncode}")
                    
                    raise MCPProtocolError("MCP server closed connection unexpectedly")
                
                line = line_bytes.decode().strip()
                logger.debug(f"📄 Decoded line ({len(line)} chars): '{line[:100]}{'...' if len(line) > 100 else ''}'")
                
                if not line:
                    logger.debug("⏭️ Empty line received, continuing...")
                    continue
                    
                # Try to parse as JSON
                try:
                    message = json.loads(line)
                    logger.debug(f"✅ Successfully parsed JSON message")
                    logger.debug(f"📨 Message type: {message.get('method', message.get('result', 'unknown'))}")
                    return message
                except json.JSONDecodeError as json_err:
                    # If not JSON, log it as informational output and continue
                    logger.info(f"ℹ️ Non-JSON output from MCP server: {line}")
                    logger.debug(f"🔍 JSON parse error: {json_err}")
                    attempts += 1
                    continue
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout waiting for message from MCP server (attempt {attempts + 1})")
                # Check if process is still alive
                if self.process.returncode is not None:
                    logger.error(f"💀 Process died during timeout - exit code: {self.process.returncode}")
                    raise MCPProtocolError(f"MCP server process died with exit code: {self.process.returncode}")
                attempts += 1
                if attempts >= max_attempts:
                    raise MCPProtocolError("Timeout waiting for message from MCP server")
                continue
            except Exception as e:
                logger.error(f"💥 Unexpected error in receive_message: {e}")
                raise MCPProtocolError(f"Failed to receive message: {e}")
        
        logger.error(f"❌ No valid JSON-RPC message received after {max_attempts} attempts")
        
        # Final check for any stderr output
        try:
            stderr_data = await asyncio.wait_for(self.process.stderr.read(), timeout=1.0)
            if stderr_data:
                stderr_msg = stderr_data.decode().strip()
                logger.error(f"📋 Final stderr check: {stderr_msg}")
        except:
            pass
            
        raise MCPProtocolError(f"No valid JSON-RPC message received after {max_attempts} attempts")

class HTTPTransport(MCPTransport):
    """HTTP+SSE transport for MCP servers"""
    
    def __init__(self, url: str):
        self.url = url
        self.client: Optional[httpx.AsyncClient] = None
        
    async def connect(self) -> None:
        """Connect to HTTP MCP server"""
        self.client = httpx.AsyncClient(timeout=30.0)
        # Test connection
        try:
            response = await self.client.get(f"{self.url}/health")
            response.raise_for_status()
            logger.info(f"Connected to HTTP MCP server: {self.url}")
        except Exception as e:
            raise MCPProtocolError(f"Failed to connect to HTTP MCP server: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from HTTP MCP server"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON-RPC message via HTTP"""
        if not self.client:
            raise MCPProtocolError("HTTP MCP server not connected")
        
        try:
            response = await self.client.post(
                f"{self.url}/message",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        except Exception as e:
            raise MCPProtocolError(f"Failed to send HTTP message: {e}")
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive JSON-RPC message via HTTP (simplified for demo)"""
        # In a real implementation, this would use SSE
        # For now, we'll use polling
        if not self.client:
            raise MCPProtocolError("HTTP MCP server not connected")
        
        try:
            response = await self.client.get(f"{self.url}/messages")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise MCPProtocolError(f"Failed to receive HTTP message: {e}")

class MCPClient:
    """MCP Client that manages connection to a single MCP server"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.transport: Optional[MCPTransport] = None
        self.request_id = 0
        self.connection_info: Optional[MCPServerConnection] = None
        
    def _get_next_request_id(self) -> int:
        """Get next request ID for JSON-RPC"""
        self.request_id += 1
        return self.request_id
    
    async def connect(self) -> MCPServerConnection:
        """Connect to the MCP server and perform handshake"""
        logger.info(f"🔗 Connecting to MCP server: {self.config.name}")
        logger.info(f"   • Transport: {self.config.transport}")
        logger.info(f"   • Command: {self.config.command}")
        logger.info(f"   • Args: {self.config.args}")
        logger.info(f"   • Environment vars: {list(self.config.env.keys()) if self.config.env else 'None'}")
        
        # Create transport
        if self.config.transport == "stdio":
            if not self.config.command:
                logger.error("❌ stdio transport requires command")
                raise MCPProtocolError("stdio transport requires command")
            logger.info(f"🚀 Creating stdio transport for: {self.config.command}")
            self.transport = StdioTransport(
                self.config.command,
                self.config.args,
                self.config.env
            )
        elif self.config.transport == "http":
            if not self.config.url:
                logger.error("❌ HTTP transport requires URL")
                raise MCPProtocolError("HTTP transport requires URL")
            logger.info(f"🌐 Creating HTTP transport for: {self.config.url}")
            self.transport = HTTPTransport(self.config.url)
        else:
            logger.error(f"❌ Unsupported transport: {self.config.transport}")
            raise MCPProtocolError(f"Unsupported transport: {self.config.transport}")
        
        # Connect transport
        logger.info(f"🔌 Connecting transport...")
        await self.transport.connect()
        logger.info(f"✅ Transport connected successfully")
        
        # Perform MCP handshake
        logger.info(f"🤝 Performing MCP handshake...")
        connection_info = await self._perform_handshake()
        logger.info(f"✅ Handshake completed")
        
        # Discover capabilities
        logger.info(f"🔍 Discovering server capabilities...")
        await self._discover_capabilities(connection_info)
        logger.info(f"✅ Capability discovery completed")
        
        self.connection_info = connection_info
        return connection_info
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self.transport:
            await self.transport.disconnect()
            self.transport = None
    
    async def _perform_handshake(self) -> MCPServerConnection:
        """Perform MCP protocol handshake"""
        logger.info(f"📋 Preparing initialize request...")
        
        # Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "docker-mcp-agent",
                    "version": "1.0.0"
                }
            }
        }
        
        logger.info(f"📤 Sending initialize request...")
        await self.transport.send_message(initialize_request)
        
        logger.info(f"📥 Waiting for initialize response...")
        response = await self.transport.receive_message()
        
        logger.info(f"📨 Received response: {response.get('method', 'response')} (id: {response.get('id', 'N/A')})")
        
        if response.get("error"):
            error_detail = response['error']
            logger.error(f"❌ Initialize failed: {error_detail}")
            raise MCPProtocolError(f"Initialize failed: {error_detail}")
        
        result = response.get("result", {})
        server_info = result.get("serverInfo", {})
        capabilities = result.get("capabilities", {})
        
        logger.info(f"🎯 Server info: {server_info}")
        logger.info(f"🔧 Server capabilities: {list(capabilities.keys())}")
        
        connection_info = MCPServerConnection(
            name=self.config.name,
            transport=self.config.transport,
            protocol_version=result.get("protocolVersion", "unknown"),
            capabilities=capabilities,
            connected=True
        )
        
        # Send initialized notification
        logger.info(f"📢 Sending initialized notification...")
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self.transport.send_message(initialized_notification)
        
        logger.info(f"✅ MCP handshake completed for {self.config.name}")
        logger.info(f"   • Protocol version: {connection_info.protocol_version}")
        logger.info(f"   • Capabilities: {list(capabilities.keys())}")
        
        return connection_info
    
    async def _discover_capabilities(self, connection_info: MCPServerConnection) -> None:
        """Discover tools, resources, and prompts from the server"""
        
        # Discover tools
        if "tools" in connection_info.capabilities:
            try:
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": self._get_next_request_id(),
                    "method": "tools/list"
                }
                await self.transport.send_message(tools_request)
                response = await self.transport.receive_message()
                
                if not response.get("error"):
                    tools_data = response.get("result", {}).get("tools", [])
                    connection_info.tools = [
                        MCPToolDefinition(**tool) for tool in tools_data
                    ]
                    logger.info(f"Discovered {len(connection_info.tools)} tools")
            except Exception as e:
                logger.warning(f"Failed to discover tools: {e}")
        
        # Discover resources
        if "resources" in connection_info.capabilities:
            try:
                resources_request = {
                    "jsonrpc": "2.0",
                    "id": self._get_next_request_id(),
                    "method": "resources/list"
                }
                await self.transport.send_message(resources_request)
                response = await self.transport.receive_message()
                
                if not response.get("error"):
                    resources_data = response.get("result", {}).get("resources", [])
                    connection_info.resources = [
                        MCPResourceDefinition(**resource) for resource in resources_data
                    ]
                    logger.info(f"Discovered {len(connection_info.resources)} resources")
            except Exception as e:
                logger.warning(f"Failed to discover resources: {e}")
        
        # Discover prompts
        if "prompts" in connection_info.capabilities:
            try:
                prompts_request = {
                    "jsonrpc": "2.0",
                    "id": self._get_next_request_id(),
                    "method": "prompts/list"
                }
                await self.transport.send_message(prompts_request)
                response = await self.transport.receive_message()
                
                if not response.get("error"):
                    prompts_data = response.get("result", {}).get("prompts", [])
                    connection_info.prompts = [
                        MCPPromptDefinition(**prompt) for prompt in prompts_data
                    ]
                    logger.info(f"Discovered {len(connection_info.prompts)} prompts")
            except Exception as e:
                logger.warning(f"Failed to discover prompts: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.transport:
            raise MCPProtocolError("Not connected to MCP server")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        await self.transport.send_message(request)
        response = await self.transport.receive_message()
        
        if response.get("error"):
            raise MCPProtocolError(f"Tool call failed: {response['error']}")
        
        return response.get("result", {})
    
    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server"""
        if not self.transport:
            raise MCPProtocolError("Not connected to MCP server")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "resources/read",
            "params": {
                "uri": resource_uri
            }
        }
        
        await self.transport.send_message(request)
        response = await self.transport.receive_message()
        
        if response.get("error"):
            raise MCPProtocolError(f"Resource read failed: {response['error']}")
        
        return response.get("result", {})
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt from the MCP server"""
        if not self.transport:
            raise MCPProtocolError("Not connected to MCP server")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "prompts/get",
            "params": {
                "name": prompt_name,
                "arguments": arguments
            }
        }
        
        await self.transport.send_message(request)
        response = await self.transport.receive_message()
        
        if response.get("error"):
            raise MCPProtocolError(f"Prompt get failed: {response['error']}")
        
        return response.get("result", {})

class MCPHost:
    """MCP Host that can manage multiple MCP server connections"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.connections: Dict[str, MCPServerConnection] = {}
    
    async def connect_to_servers(self, server_configs: List[MCPServerConfig]) -> List[MCPServerConnection]:
        """Connect to multiple MCP servers"""
        connections = []
        
        for config in server_configs:
            try:
                client = MCPClient(config)
                connection = await client.connect()
                
                self.clients[config.name] = client
                self.connections[config.name] = connection
                connections.append(connection)
                
                logger.info(f"Connected to MCP server: {config.name}")
            except Exception as e:
                logger.error(f"Failed to connect to {config.name}: {e}")
                # Create error connection info
                error_connection = MCPServerConnection(
                    name=config.name,
                    transport=config.transport,
                    protocol_version="unknown",
                    connected=False,
                    error=str(e)
                )
                connections.append(error_connection)
        
        return connections
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        for client in self.clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting client: {e}")
        
        self.clients.clear()
        self.connections.clear()
    
    def get_all_tools(self) -> Dict[str, List[MCPToolDefinition]]:
        """Get all available tools from all connected servers"""
        tools = {}
        for server_name, connection in self.connections.items():
            if connection.connected:
                tools[server_name] = connection.tools
        return tools
    
    def get_all_resources(self) -> Dict[str, List[MCPResourceDefinition]]:
        """Get all available resources from all connected servers"""
        resources = {}
        for server_name, connection in self.connections.items():
            if connection.connected:
                resources[server_name] = connection.resources
        return resources
    
    def get_all_prompts(self) -> Dict[str, List[MCPPromptDefinition]]:
        """Get all available prompts from all connected servers"""
        prompts = {}
        for server_name, connection in self.connections.items():
            if connection.connected:
                prompts[server_name] = connection.prompts
        return prompts
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific server"""
        if server_name not in self.clients:
            raise ValueError(f"Server {server_name} not connected")
        
        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)
    
    async def read_resource(self, server_name: str, resource_uri: str) -> Dict[str, Any]:
        """Read a resource from a specific server"""
        if server_name not in self.clients:
            raise ValueError(f"Server {server_name} not connected")
        
        client = self.clients[server_name]
        return await client.read_resource(resource_uri)
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt from a specific server"""
        if server_name not in self.clients:
            raise ValueError(f"Server {server_name} not connected")
        
        client = self.clients[server_name]
        return await client.get_prompt(prompt_name, arguments) 