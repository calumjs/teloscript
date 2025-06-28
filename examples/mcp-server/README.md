# TELOSCRIPT MCP Server

TELOSCRIPT can be run as an MCP (Model Context Protocol) server, enabling recursive agent orchestration where agents can spawn and manage other agents.

## Overview

The TELOSCRIPT MCP Server exposes TELOSCRIPT's functionality as MCP tools that can be used by other agents or MCP clients. This creates a powerful recursive architecture where intelligence can be composed and scaled infinitely.

## Features

### Core Tools

- **`spawn_agent`** - Create and execute a new TELOSCRIPT agent
- **`check_agent_status`** - Check the status of a running agent
- **`execute_purpose_endpoint`** - Execute a predefined purpose endpoint
- **`list_purpose_endpoints`** - List all available purpose endpoints
- **`get_agent_capabilities`** - List available MCP servers and their capabilities
- **`create_workflow_template`** - Create a reusable workflow template
- **`list_active_agents`** - List all currently active agents
- **`cancel_agent`** - Cancel a running agent

## Quick Start

### 1. Start TELOSCRIPT API Server

First, ensure the main TELOSCRIPT API is running:

```bash
python main.py
```

This starts the API server on http://localhost:8000

### 2. Run as MCP Server

#### Option A: Standalone Script

```bash
python teloscript_mcp_server.py
```

#### Option B: With Custom API URL

```bash
python teloscript_mcp_server.py --api-url http://localhost:8000
```

### 3. Configure in MCP Client

Add TELOSCRIPT to your MCP client configuration:

```json
{
  "name": "teloscript",
  "command": "python",
  "args": ["/path/to/teloscript_mcp_server.py"],
  "transport": "stdio"
}
```

## Usage Examples

### Example 1: Spawn a Research Agent

```python
# Using TELOSCRIPT to spawn a research agent
result = await mcp_client.call_tool("spawn_agent", {
    "goal": "Research the latest AI developments and create a summary report",
    "mcp_servers": ["brave-search", "filesystem"],
    "max_iterations": 15,
    "timeout": 300
})

print(f"Agent spawned: {result['agent_id']}")
```

### Example 2: Chain Multiple Operations

```python
# Spawn research agent
research_result = await mcp_client.call_tool("spawn_agent", {
    "goal": "Research Python async patterns and save findings to research.md"
})

research_agent_id = research_result["agent_id"]

# Wait for completion
while True:
    status = await mcp_client.call_tool("check_agent_status", {
        "agent_id": research_agent_id
    })
    
    if status["status_data"]["status"] == "completed":
        break
    
    await asyncio.sleep(2)

# Spawn analysis agent using the research
analysis_result = await mcp_client.call_tool("spawn_agent", {
    "goal": "Read research.md and create implementation recommendations",
    "mcp_servers": ["filesystem"]
})
```

### Example 3: Execute Purpose Endpoint

```python
# Execute a predefined purpose endpoint
result = await mcp_client.call_tool("execute_purpose_endpoint", {
    "endpoint_slug": "handle-github-webhook",
    "input_data": {
        "repository": "user/awesome-project",
        "action": "push",
        "branch": "main"
    }
})
```

### Example 4: Get Capabilities

```python
# List available MCP servers and capabilities
capabilities = await mcp_client.call_tool("get_agent_capabilities", {})
print("Available MCP servers:", capabilities["mcp_servers"])
```

## Tool Reference

### spawn_agent

Creates and executes a new TELOSCRIPT agent.

**Parameters:**
- `goal` (string, required): Goal for the agent to accomplish
- `mcp_servers` (array, optional): List of MCP server names to use (default: ["filesystem"])
- `max_iterations` (integer, optional): Maximum iterations for goal completion (default: 10)
- `timeout` (integer, optional): Overall timeout in seconds (default: 300)

**Returns:**
```json
{
  "success": true,
  "agent_id": "uuid-string",
  "status": "completed",
  "message": "Agent spawned successfully",
  "result": { ... }
}
```

### check_agent_status

Checks the status of a running agent.

**Parameters:**
- `agent_id` (string, required): ID of the agent to check

**Returns:**
```json
{
  "success": true,
  "agent_id": "uuid-string",
  "status_data": {
    "status": "running",
    "progress": 45.5,
    "current_activity": "Analyzing data...",
    ...
  }
}
```

### execute_purpose_endpoint

Executes a predefined purpose endpoint.

**Parameters:**
- `endpoint_slug` (string, required): Slug of the purpose endpoint
- `input_data` (object, required): Input data for the endpoint

**Returns:**
```json
{
  "success": true,
  "endpoint_slug": "handle-github-webhook",
  "result": { ... }
}
```

## Recursive Architecture Benefits

1. **Composable Intelligence**: Agents can orchestrate other agents for complex tasks
2. **Specialization**: Different TELOSCRIPT instances can specialize in different domains
3. **Scalability**: Distribute workload across multiple TELOSCRIPT instances
4. **Modularity**: Build complex workflows from simple, reusable components

## Configuration

### Environment Variables

- `TELOSCRIPT_API_URL`: URL of the TELOSCRIPT API (default: http://localhost:8000)

### MCP Server Configuration

When configuring TELOSCRIPT as an MCP server in your client:

```json
{
  "name": "teloscript",
  "command": "python",
  "args": ["/absolute/path/to/teloscript_mcp_server.py", "--api-url", "http://localhost:8000"],
  "transport": "stdio",
  "env": {
    "TELOSCRIPT_API_URL": "http://localhost:8000"
  }
}
```

## Error Handling

All tools return a consistent error format:

```json
{
  "success": false,
  "error": "Error message",
  "tool": "tool_name",
  "arguments": { ... }
}
```

## Integration with Existing TELOSCRIPT Features

The MCP server integrates seamlessly with existing TELOSCRIPT features:

- **Purpose Endpoints**: Execute predefined workflows via `execute_purpose_endpoint`
- **MCP Servers**: All configured MCP servers are available to spawned agents
- **Dashboard**: Spawned agents appear in the TELOSCRIPT dashboard
- **Status Tracking**: Real-time status updates for all operations

## Advanced Usage

### Creating Workflow Templates

```python
template = await mcp_client.call_tool("create_workflow_template", {
    "name": "Research and Analyze",
    "description": "Research a topic and create analysis",
    "steps": [
        {
            "tool": "spawn_agent",
            "goal": "Research {{topic}} and save findings",
            "servers": ["brave-search", "filesystem"]
        },
        {
            "tool": "spawn_agent", 
            "goal": "Analyze research findings and create report",
            "servers": ["filesystem"]
        }
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"}
        }
    }
})
```

This creates a powerful recursive architecture where intelligence can be composed and scaled infinitely!