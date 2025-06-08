# Docker MCP Agent System
__version__ = "1.0.0"

from .models import (
    AgentStatus,
    MCPServerConfig,
    MCPAgentConfig,
    AgentGoal,
    AgentRequest,
    StatusUpdate,
    AgentResponse,
    MCPToolDefinition,
    MCPResourceDefinition,
    MCPPromptDefinition,
    MCPServerConnection
)

from .mcp_host import MCPHost, MCPClient
from .mcp_agent import MCPAgent
from .orchestrator import WorkflowOrchestrator

__all__ = [
    "AgentStatus",
    "MCPServerConfig",
    "MCPAgentConfig", 
    "AgentGoal",
    "AgentRequest",
    "StatusUpdate",
    "AgentResponse",
    "MCPToolDefinition",
    "MCPResourceDefinition",
    "MCPPromptDefinition",
    "MCPServerConnection",
    "MCPHost",
    "MCPClient",
    "MCPAgent",
    "WorkflowOrchestrator"
] 