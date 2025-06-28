# 📦 TELOSCRIPT MCP Server Installation Guide

**Run TELOSCRIPT as an MCP Server - Like NPX for Python!**

## 🚀 Quick Install & Run

### Option 1: Install Globally (Recommended)

```bash
# Install from the local directory
pip install -e .

# Or install from PyPI (when published)
pip install teloscript-mcp

# Run the MCP server
teloscript-mcp
```

### Option 2: Install in Virtual Environment

```bash
# Create virtual environment
python -m venv teloscript-mcp-env
source teloscript-mcp-env/bin/activate  # On Windows: teloscript-mcp-env\Scripts\activate

# Install
pip install -e .

# Run
teloscript-mcp
```

### Option 3: Direct Python Execution

```bash
# Run directly without installation
python src/teloscript_mcp.py
```

## 🛠️ Command Line Usage

### Basic Usage

```bash
# Start with default settings
teloscript-mcp

# Specify custom API URL
teloscript-mcp --api-url http://localhost:8000

# Enable debug logging
teloscript-mcp --log-level DEBUG

# Show help
teloscript-mcp --help
```

### Command Line Options

```
usage: teloscript-mcp [-h] [--api-url API_URL] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--version]

TELOSCRIPT MCP Server - Enable recursive agent orchestration through MCP protocol

options:
  -h, --help            show this help message and exit
  --api-url API_URL     URL of the TELOSCRIPT API (default: http://localhost:8000)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set the logging level (default: INFO)
  --version             show program's version number and exit
```

## 📋 Prerequisites

### 1. Start TELOSCRIPT API

Before running the MCP server, ensure the TELOSCRIPT API is running:

```bash
# In the TELOSCRIPT directory
python main.py
```

The API should be accessible at `http://localhost:8000` (or your custom URL).

### 2. Install Dependencies

If installing manually:

```bash
pip install mcp>=1.1.0 httpx>=0.27.0
```

## 🔧 MCP Client Configuration

### Claude Desktop Configuration

Add to your Claude Desktop MCP configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "teloscript": {
      "command": "teloscript-mcp",
      "args": ["--api-url", "http://localhost:8000"],
      "env": {}
    }
  }
}
```

### Generic MCP Client Configuration

```json
{
  "name": "teloscript",
  "command": "teloscript-mcp",
  "args": ["--api-url", "http://localhost:8000"],
  "transport": "stdio"
}
```

### Multiple TELOSCRIPT Instances

Configure multiple specialized TELOSCRIPT instances:

```json
{
  "mcpServers": {
    "teloscript-main": {
      "command": "teloscript-mcp",
      "args": ["--api-url", "http://localhost:8000"]
    },
    "teloscript-research": {
      "command": "teloscript-mcp", 
      "args": ["--api-url", "http://research-server:8001"]
    },
    "teloscript-dev": {
      "command": "teloscript-mcp",
      "args": ["--api-url", "http://dev-server:8002"]
    }
  }
}
```

## 🧪 Testing the Installation

### 1. Test Command Installation

```bash
# Check if command is available
which teloscript-mcp

# Test help output
teloscript-mcp --help

# Test version
teloscript-mcp --version
```

### 2. Test MCP Server Connection

```bash
# Start in one terminal
teloscript-mcp --log-level DEBUG

# The server should show:
# INFO - Starting TELOSCRIPT MCP Server...
# INFO - Connecting to TELOSCRIPT API at: http://localhost:8000
# INFO - Server ready for MCP connections via stdio
```

### 3. Test with MCP Client

Once connected to an MCP client, you should see these tools available:

- ✅ `spawn_agent`
- ✅ `check_agent_status`
- ✅ `execute_purpose_endpoint`
- ✅ `list_purpose_endpoints`
- ✅ `get_agent_capabilities`
- ✅ `list_active_agents`
- ✅ `cancel_agent`
- ✅ `create_workflow_template`

## 🔄 Development Installation

For development and contributing:

```bash
# Clone repository
git clone https://github.com/calumjs/teloscript.git
cd teloscript

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/
```

## 🌍 Environment Variables

You can also configure using environment variables:

```bash
export TELOSCRIPT_API_URL=http://localhost:8000
export TELOSCRIPT_LOG_LEVEL=INFO

# Then run
teloscript-mcp
```

## 📦 Distribution

### Build Package

```bash
# Build wheel and source distribution
python -m build

# Install built package
pip install dist/teloscript_mcp-1.0.0-py3-none-any.whl
```

### Publish to PyPI

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*
```

## 🚨 Troubleshooting

### Common Issues

**1. Command not found**
```bash
# Check if pip install worked
pip list | grep teloscript-mcp

# Reinstall if needed
pip install -e . --force-reinstall
```

**2. Connection refused**
```bash
# Check if TELOSCRIPT API is running
curl http://localhost:8000/health

# Start TELOSCRIPT API
python main.py
```

**3. Import errors**
```bash
# Check MCP installation
pip install mcp>=1.1.0 --upgrade

# Check Python version
python --version  # Should be 3.8+
```

**4. Permission errors on Windows**
```bash
# Run as administrator or use virtual environment
python -m venv venv
venv\Scripts\activate
pip install -e .
```

### Debug Mode

Run with maximum logging:

```bash
teloscript-mcp --log-level DEBUG --api-url http://localhost:8000
```

### Logs Location

The MCP server logs to stdout/stderr by default. To save logs:

```bash
# Save logs to file
teloscript-mcp --log-level DEBUG 2>&1 | tee teloscript-mcp.log
```

## 🎯 Usage Examples

### Example 1: Basic Agent Spawning

After connecting to Claude Desktop:

```
I want you to use the teloscript tools to spawn an agent that researches "latest AI developments" and saves the findings to a file.
```

Claude will use the `spawn_agent` tool:
```json
{
  "goal": "Research latest AI developments and save findings to ai_research.md",
  "mcp_servers": ["brave-search", "filesystem"],
  "max_iterations": 15
}
```

### Example 2: Checking Agent Status

```
Check the status of agent ID abc-123-def
```

Claude will use `check_agent_status`:
```json
{
  "agent_id": "abc-123-def"
}
```

### Example 3: Recursive Orchestration

```
Spawn a master agent that coordinates multiple sub-agents to create a complete web development project plan.
```

Claude will use `spawn_agent` with recursive capabilities:
```json
{
  "goal": "Master orchestrator: spawn planning agent, architecture agent, and documentation agent to create comprehensive web dev plan",
  "mcp_servers": ["teloscript", "filesystem", "github"],
  "max_iterations": 25,
  "timeout": 600
}
```

---

**🎉 You're now ready to use TELOSCRIPT as an MCP server for recursive agent orchestration!**