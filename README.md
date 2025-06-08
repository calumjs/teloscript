# 🎯 TELOSCRIPT
**Purposeful Agent Orchestration System**

> *"He who has a why to live can bear almost any how." - Friedrich Nietzsche*

TELOSCRIPT is a sophisticated agent orchestration platform that coordinates  MCP (Model Context Protocol) servers toward any goal you can think of. Unlike chat-based MCP implementations, TELOSCRIPT empowers fully autonomous goal resolution by seamlessly orchestrating your provided MCP servers into a purposeful, coordinated system.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Node.js 18+ (for MCP servers)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/calumjs/teloscript
cd teloscript
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the system**
```bash
# Development mode
python main.py

# Production with Docker
docker-compose up -d
```

4. **Access the orchestration interface**
- Primary Interface: http://localhost:8000/dashboard/test
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

The web interface provides real-time agent monitoring, configuration management, and an intuitive way to launch agents with visual feedback.

## 🎯 Usage Examples

### Simple Goal Execution
```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: text/plain" \
  -d "Write a Nietzschean aphorism about automating the achievement of goals"
```

### Advanced Configuration with Multiple Servers
```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Research Python async patterns and save findings to a report",
    "servers": [
      {
        "name": "brave-search",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "your-api-key"},
        "transport": "stdio"
      },
      {
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "transport": "stdio"
      }
    ],
    "max_iterations": 15,
    "timeout": 300
  }'
```

## 🏛️ Architecture

TELOSCRIPT implements a **distributed coordination** model where agents maintain autonomy while working toward shared objectives:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Agent A   │    │   MCP Agent B   │    │   MCP Agent C   │
│   (Filesystem)  │    │  (Web Search)   │    │    (GitHub)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────┬──────────────────────────────┘
                         │
                ┌─────────▼───────┐
                │   TELOSCRIPT    │
                │  Orchestrator   │
                │  (Coordination  │
                │    Engine)      │
                └─────────────────┘
```

### Component Overview

- **Orchestrator**: Central coordination engine that manages agent interactions
- **MCP Agents**: Specialized agents handling specific domains (files, web, APIs, etc.)
- **Orchestration Interface**: Real-time monitoring and control interface  
- **API Gateway**: RESTful interface for external integrations
- **Configuration Manager**: Dynamic agent and system configuration

## 📁 Project Structure

```
teloscript/
├── src/
│   ├── api.py              # FastAPI application & orchestration endpoints
│   ├── orchestrator.py     # Agent coordination and goal management
│   ├── mcp_agent.py        # Individual agent implementation
│   ├── models.py           # Data models & API schemas
│   └── utils/              # Utility functions and helpers
├── config/
│   ├── mcp_configs.json    # MCP server configurations
│   └── system_config.yaml  # System-wide settings
├── scripts/
│   ├── startup.sh          # System initialization
│   └── preload-mcps.sh     # MCP server preloading
├── docker-compose.yml      # Container orchestration
├── Dockerfile             # Container definition
├── nginx.conf             # Reverse proxy configuration
├── requirements.txt       # Python dependencies
└── main.py               # Application entry point
```

## ⚙️ Configuration

### MCP Server Configuration
TELOSCRIPT uses a flexible configuration system for MCP servers:

```json
{
  "filesystem": {
    "name": "Filesystem Server",
    "description": "Access and manipulate files in the project directory",
    "config": {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "transport": "stdio"
    },
    "capabilities": ["read", "write", "search", "monitor"]
  },
  "brave-search": {
    "name": "Brave Search",
    "description": "Web search capabilities using Brave Search API",
    "config": {
      "name": "brave-search",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "transport": "stdio"
    },
    "capabilities": ["search", "summarize"],
    "requires_api_key": true
  }
}
```

### Environment Variables

You only need to specify your OpenAI API key to use Teloscript orchestrator - obviously individual MCP servers may require their own configuration.

```bash
OPENAI_API_KEY=your-open-ai-api-key
```

## 🔌 API Reference

### Core Agent Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/agents` | Launch agent with goal (text or JSON) |
| `POST` | `/agents/launch` | Launch agent with UI-selected MCP configs |
| `GET` | `/agents/{id}/status` | Get agent execution status |
| `GET` | `/agents/{id}/stream` | Stream real-time agent updates |
| `DELETE` | `/agents/{id}` | Cancel running agent |
| `DELETE` | `/agents` | Cancel all running agents |
| `GET` | `/agents` | List all active agents |

### Configuration Management

| Method | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/mcp-configs` | List all MCP configurations |
| `GET` | `/mcp-configs/{id}` | Get specific MCP configuration |
| `GET` | `/mcp-configs/info` | Get configuration file information |
| `POST` | `/mcp-configs` | Create new MCP configuration |
| `PUT` | `/mcp-configs/{id}` | Update existing configuration |
| `DELETE` | `/mcp-configs/{id}` | Remove configuration |
| `POST` | `/mcp-configs/reload` | Reload configurations from file |

### Dashboard & Health

| Method | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/dashboard` | Get orchestration state |
| `GET` | `/dashboard/stream` | Stream real-time orchestration updates |
| `GET` | `/dashboard/test` | Web orchestration interface |
| `GET` | `/health` | System health check |
| `GET` | `/` | API information |

### Examples & Documentation

| Method | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/examples/mcp-config` | Example MCP configurations |
| `GET` | `/examples/request` | Example agent request format |

## 🐳 Docker Deployment

### Development Environment
```bash
# Start all services
docker-compose up

# Watch logs
docker-compose logs -f teloscript
```

### Production Environment
```bash
# Production deployment with scaling
docker-compose --profile production up -d

# Scale specific services
docker-compose up -d --scale teloscript=3
```

The production profile includes:
- Nginx reverse proxy with SSL termination
- Optimized container configurations
- Health checks and automatic restart policies  
- Persistent logging and configuration storage
- Resource limits and monitoring

## 🔒 Security Considerations

- **Container Security**: Runs as non-privileged user with minimal permissions
- **Network Isolation**: Services communicate through isolated Docker networks  
- **Input Validation**: All API inputs are validated and sanitized
- **API Rate Limiting**: Built-in rate limiting to prevent abuse
- **Secret Management**: Environment variables for sensitive configuration
- **Audit Logging**: Comprehensive logging of all agent activities

## 🤝 Contributing

We welcome contributions that align with TELOSCRIPT's philosophy of purposeful technology:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes with clear, purposeful messages
4. **Test** your changes thoroughly
5. **Push** to your branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request with detailed description

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Document new functionality clearly
- Ensure Docker compatibility

## 🛠️ Troubleshooting

### Common Issues

**Agents Not Starting**
- Check MCP server configurations in `config/mcp_configs.json`
- Verify Node.js and npm are installed
- Check system logs: `docker-compose logs teloscript`

**Orchestration Interface Not Loading**
- Ensure port 8000 is available
- Check nginx configuration
- Verify WebSocket connections are allowed

## 🙏 Acknowledgments

- **Manus**: For the original idea - see [Manus.im](https://manus.im)
- **Model Context Protocol (MCP)**: Foundation for agent communication standards
- **FastAPI**: Modern, fast Python web framework powering our API
- **Docker**: Containerization platform enabling seamless deployment

---

**TELOSCRIPT** - *Where purpose meets autonomous coordination*