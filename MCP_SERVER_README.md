# TELOSCRIPT MCP Server

Run TELOSCRIPT as an MCP server for recursive agent orchestration.

## Usage

```bash
# Install and run with uvx (recommended)
uvx teloscript-mcp

# Or install locally
pip install -e .
teloscript-mcp
```

## Configuration

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "teloscript": {
      "command": "uvx",
      "args": ["teloscript-mcp"]
    }
  }
}
```

## Options

```bash
uvx teloscript-mcp --api-url http://localhost:8000 --log-level DEBUG
```

## Tools

- `spawn_agent` - Create agents with goals and MCP servers
- `check_agent_status` - Monitor agent progress  
- `execute_purpose_endpoint` - Run predefined workflows
- `list_purpose_endpoints` - List available endpoints
- `get_agent_capabilities` - List MCP server capabilities
- `list_active_agents` - View active agents
- `cancel_agent` - Stop agents
- `create_workflow_template` - Build reusable workflows