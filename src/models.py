from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import uuid
from datetime import datetime

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MCPToolDefinition(BaseModel):
    """MCP Tool definition following the specification"""
    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for tool input")

class MCPResourceDefinition(BaseModel):
    """MCP Resource definition following the specification"""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

class MCPPromptDefinition(BaseModel):
    """MCP Prompt definition following the specification"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server"""
    name: str
    command: Optional[str] = None  # For stdio transport
    args: Optional[List[str]] = Field(default_factory=list)  # Command arguments
    env: Optional[Dict[str, str]] = Field(default_factory=dict)  # Environment variables
    url: Optional[str] = None  # For HTTP transport
    transport: str = Field(default="stdio", description="Transport type: stdio or http")
    timeout: int = Field(default=30, description="Connection timeout in seconds")

class AgentGoal(BaseModel):
    """Goal definition for the MCP agent"""
    description: str
    success_criteria: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: int = Field(default=1, description="Priority level (1-10)")
    
class MCPAgentConfig(BaseModel):
    """Configuration for an MCP agent that can connect to multiple servers"""
    servers: List[MCPServerConfig]
    goal: AgentGoal
    max_iterations: int = Field(default=10, description="Maximum iterations for goal completion")
    timeout: int = Field(default=300, description="Overall timeout in seconds")

class AgentRequest(BaseModel):
    """Request to create and run an MCP agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: MCPAgentConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StatusUpdate(BaseModel):
    """Status update from a running agent"""
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: AgentStatus
    message: str
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MCPServerConnection(BaseModel):
    """Information about a connected MCP server"""
    name: str
    transport: str
    protocol_version: str
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    tools: List[MCPToolDefinition] = Field(default_factory=list)
    resources: List[MCPResourceDefinition] = Field(default_factory=list)
    prompts: List[MCPPromptDefinition] = Field(default_factory=list)
    connected: bool = False
    error: Optional[str] = None

class AgentResponse(BaseModel):
    """Final response from an MCP agent"""
    agent_id: str
    status: AgentStatus
    result: str
    execution_time: float
    iterations_used: int
    status_updates: List[StatusUpdate]
    server_connections: List[MCPServerConnection] = Field(default_factory=list)
    error: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.utcnow)

class ToolInvocation(BaseModel):
    """Represents a tool call to an MCP server"""
    server_name: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ResourceRequest(BaseModel):
    """Represents a resource request to an MCP server"""
    server_name: str
    resource_uri: str

class PromptInvocation(BaseModel):
    """Represents a prompt invocation on an MCP server"""
    server_name: str
    prompt_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

# Purpose Endpoint Models
class PurposeEndpointConfig(BaseModel):
    """Configuration for a purpose endpoint"""
    slug: str = Field(description="Unique identifier for the endpoint (e.g., 'handle-github-webhook')")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Purpose and functionality description")
    prompt_template: str = Field(description="Template prompt that will be used for all requests")
    mcp_servers: List[str] = Field(description="List of MCP server names to use")
    max_iterations: int = Field(default=10, description="Maximum iterations for goal completion")
    timeout: int = Field(default=300, description="Overall timeout in seconds")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    enabled: bool = Field(default=True, description="Whether the endpoint is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PurposeEndpointRequest(BaseModel):
    """Request to execute a purpose endpoint"""
    endpoint_slug: str = Field(description="The slug of the purpose endpoint to execute")
    input_data: Union[Dict[str, Any], str, Any] = Field(description="Input data for the endpoint (any format)")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class PurposeEndpointResponse(BaseModel):
    """Response from a purpose endpoint execution"""
    request_id: str
    endpoint_slug: str
    status: AgentStatus
    result: str
    execution_time: float
    iterations_used: int
    input_data: Union[Dict[str, Any], str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: datetime = Field(default_factory=datetime.utcnow) 