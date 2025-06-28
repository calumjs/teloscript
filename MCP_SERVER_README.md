# 🔌 TELOSCRIPT MCP Server

**Recursive Agent Orchestration through Model Context Protocol**

TELOSCRIPT can now operate as an MCP (Model Context Protocol) server, enabling powerful recursive agent orchestration where agents can spawn and manage other agents.

## 🌟 Overview

This feature transforms TELOSCRIPT into an MCP server that exposes its functionality as standardized MCP tools. This creates a revolutionary recursive architecture where intelligence can be composed and scaled infinitely!

### Key Benefits

- **🔄 Recursive Architecture**: Agents can orchestrate other agents for complex tasks
- **🎯 Specialization**: Different TELOSCRIPT instances can specialize in different domains
- **📈 Scalability**: Distribute workload across multiple TELOSCRIPT instances
- **🧩 Modularity**: Build complex workflows from simple, reusable components

## 🚀 Quick Start

### 1. Install TELOSCRIPT MCP Server

```bash
# Install globally (like NPX)
pip install -e .

# Or install from PyPI (when published)
pip install teloscript-mcp
```

### 2. Start TELOSCRIPT API

```bash
python main.py
```

### 3. Run as MCP Server

```bash
# Option A: Default configuration
teloscript-mcp

# Option B: Custom API URL
teloscript-mcp --api-url http://localhost:8000

# Option C: With debug logging
teloscript-mcp --log-level DEBUG
```

### 4. Configure in MCP Client

Add to your MCP client configuration:

```json
{
  "name": "teloscript",
  "command": "teloscript-mcp",
  "args": ["--api-url", "http://localhost:8000"],
  "transport": "stdio"
}
```

## 🛠️ Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `spawn_agent` | Create and execute a new TELOSCRIPT agent | `goal`, `mcp_servers`, `max_iterations`, `timeout` |
| `check_agent_status` | Check the status of a running agent | `agent_id` |
| `execute_purpose_endpoint` | Execute a predefined purpose endpoint | `endpoint_slug`, `input_data` |
| `list_purpose_endpoints` | List all available purpose endpoints | None |
| `get_agent_capabilities` | List available MCP servers and capabilities | None |
| `create_workflow_template` | Create a reusable workflow template | `name`, `description`, `steps`, `input_schema` |
| `list_active_agents` | List all currently active agents | None |
| `cancel_agent` | Cancel a running agent | `agent_id` |

## 💡 Usage Examples

### Example 1: Simple Research Agent

```python
result = await mcp_client.call_tool("spawn_agent", {
    "goal": "Research the latest AI developments and create a summary report",
    "mcp_servers": ["brave-search", "filesystem"],
    "max_iterations": 15,
    "timeout": 300
})
```

### Example 2: Chained Operations

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

# Spawn analysis agent
analysis_result = await mcp_client.call_tool("spawn_agent", {
    "goal": "Read research.md and create implementation recommendations",
    "mcp_servers": ["filesystem"]
})
```

### Example 3: Recursive Orchestration

```python
# Master agent that manages sub-agents
master_goal = """
You are a master orchestrator. Your task is to coordinate multiple specialized agents:

1. Spawn a research agent to gather information about "Python web frameworks"
2. Spawn another agent to analyze the research and create a comparison
3. Spawn a final agent to create a recommendation report

Use the TELOSCRIPT MCP tools available to you to spawn and manage these agents.
"""

master_result = await mcp_client.call_tool("spawn_agent", {
    "goal": master_goal,
    "mcp_servers": ["teloscript", "filesystem"],  # Include teloscript!
    "max_iterations": 20,
    "timeout": 600
})
```

## 📁 File Structure

```
├── src/
│   └── mcp_server.py              # Main MCP server implementation
├── examples/
│   └── mcp-server/
│       ├── README.md              # Detailed documentation
│       ├── client_example.py      # Example MCP client usage
│       └── mcp_config_example.json # Configuration examples
├── teloscript_mcp_server.py       # Standalone server script
└── test_mcp_server.py            # Test suite
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_mcp_server.py
```

Expected output:
```
TELOSCRIPT MCP Server Tests
===========================
✓ All tests passed! MCP server implementation is working correctly.

🎉 TELOSCRIPT MCP Server is ready for use!
```

## ⚙️ Configuration

### Environment Variables

- `TELOSCRIPT_API_URL`: URL of the TELOSCRIPT API (default: http://localhost:8000)

### MCP Client Configuration

```json
{
  "mcpServers": {
    "teloscript": {
      "command": "python",
      "args": ["/absolute/path/to/teloscript_mcp_server.py"],
      "transport": "stdio",
      "env": {
        "TELOSCRIPT_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

## 🔗 Integration

The MCP server integrates seamlessly with existing TELOSCRIPT features:

- **Purpose Endpoints**: Execute predefined workflows via `execute_purpose_endpoint`
- **MCP Servers**: All configured MCP servers are available to spawned agents
- **Dashboard**: Spawned agents appear in the TELOSCRIPT dashboard
- **Status Tracking**: Real-time status updates for all operations

## 🎯 Use Cases

### 1. **Multi-Agent Research Pipeline**
- Research agent gathers information
- Analysis agent processes findings
- Report agent creates final output

### 2. **Code Development Workflow**
- Planning agent creates specifications
- Coding agent implements features
- Testing agent validates functionality
- Documentation agent creates docs

### 3. **Content Creation Pipeline**
- Research agent gathers source material
- Writing agent creates content
- Review agent checks quality
- Publishing agent formats and distributes

### 4. **Data Processing Workflow**
- Extraction agent gathers data
- Cleaning agent processes data
- Analysis agent finds insights
- Visualization agent creates charts

## 🚦 API Endpoints

The main TELOSCRIPT API also includes new endpoints:

- `POST /mcp/server/start` - Get instructions for starting MCP server mode
- `GET /mcp/server/info` - Get information about MCP server capabilities

## 🔧 Advanced Usage

### Custom Server Configurations

The MCP server supports all standard MCP server configurations including:

- Custom command arguments
- Environment variables
- Transport configuration
- Timeout settings

### Error Handling

All tools return consistent error responses:

```json
{
  "success": false,
  "error": "Error message",
  "tool": "tool_name",
  "arguments": { ... }
}
```

## 📈 Performance

- Lightweight MCP protocol implementation
- Efficient tool routing and execution
- Parallel agent support
- Resource-conscious design

## 🔮 Future Enhancements

- HTTP transport support
- Advanced workflow templates
- Agent clustering and load balancing
- Enhanced monitoring and metrics
- Cross-instance agent communication

## 🤝 Contributing

See the main TELOSCRIPT README for contribution guidelines. When contributing to the MCP server feature:

1. Run the test suite: `python test_mcp_server.py`
2. Test with actual MCP clients
3. Update documentation as needed

## 📝 License

Same as TELOSCRIPT main project.

---

**Transform your AI workflows with recursive agent orchestration! 🚀**