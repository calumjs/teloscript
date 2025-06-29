# 🎯 TELOSCRIPT v1.1.0 Release Notes

**Release Date**: June 23, 2025  
**Tag**: `v1.1.0`

## 🚀 Major New Features

### Purpose Endpoints System
The biggest addition in v1.1.0 is the **Purpose Endpoints** system - a revolutionary way to create reusable, predefined agent workflows.

#### What are Purpose Endpoints?
Purpose Endpoints combine prompts, MCP servers, and execution parameters into single, callable endpoints. Instead of configuring agents manually each time, you can now:

```bash
POST /purpose/youtube-video-to-blog
{
  "url": "https://www.youtube.com/watch?v=example"
}
```

This automatically:
- Uses the YouTube transcript MCP server
- Applies a specialized blog-writing prompt
- Saves the result to your filesystem
- All in one streamlined call

#### Built-in Purpose Endpoints
v1.1.0 ships with several ready-to-use purpose endpoints:

- **`youtube-video-to-blog`** - Convert YouTube videos to blog posts
- **`handle-github-webhook`** - Process GitHub webhooks automatically  
- **`analyze-code-changes`** - Analyze code changes and provide insights
- **`research-topic`** - Research topics using web search and document findings

#### Purpose Endpoint Management API
Complete CRUD operations for managing purpose endpoints:

```bash
# List all endpoints
GET /purpose/endpoints

# Create new endpoint
POST /purpose/endpoints

# Update existing endpoint  
PUT /purpose/endpoints/{slug}

# Delete endpoint
DELETE /purpose/endpoints/{slug}

# Execute endpoint
POST /purpose/{slug}
```

### YouTube Integration
Full YouTube video processing capabilities:

- **YouTube Transcript MCP Server** - Extract transcripts from any YouTube video
- **Automatic Blog Generation** - Convert video content to structured blog posts
- **uvx Package Manager Support** - Enhanced Python package management for MCP servers

### Enhanced MCP Server Ecosystem
Expanded support for MCP servers:

- **YouTube Transcript Server** - `uvx --from git+https://github.com/jkawamoto/mcp-youtube-transcript`
- **Tavily Search Integration** - Advanced web search capabilities
- **FAISS Vector Search** - Semantic search and similarity matching
- **Playwright Web Automation** - Advanced browser automation

## 🔧 Technical Improvements

### Windows Compatibility Enhancements
- **uvx Command Support** - Proper handling of uvx commands on Windows
- **Environment Variable Management** - Enhanced PATH resolution for subprocess execution
- **PowerShell Integration** - Better PowerShell command execution

### Configuration Management
- **Dynamic MCP Configuration** - Runtime MCP server configuration management
- **Purpose Endpoint Configuration** - JSON-based endpoint definitions
- **Environment-Specific Settings** - Better separation of dev/prod configurations

### Error Handling & Logging
- **Enhanced Error Messages** - More descriptive error reporting
- **Structured Logging** - Better log formatting and categorization
- **Process Monitoring** - Improved subprocess lifecycle management

## 📡 API Enhancements

### New Endpoints
```bash
# Purpose Endpoint Management
GET    /purpose/endpoints              # List all purpose endpoints
POST   /purpose/endpoints              # Create new purpose endpoint
GET    /purpose/endpoints/{slug}       # Get specific purpose endpoint
PUT    /purpose/endpoints/{slug}       # Update purpose endpoint
DELETE /purpose/endpoints/{slug}       # Delete purpose endpoint

# Purpose Endpoint Execution
POST   /purpose/{slug}                 # Execute purpose endpoint
POST   /purpose/{slug}/stream         # Execute with streaming updates

# Enhanced Configuration
POST   /mcp-configs/reload            # Reload MCP configurations
GET    /mcp-configs/info              # Get configuration file info
```

### Response Format Improvements
- **Structured Error Responses** - Consistent error formatting
- **Enhanced Status Updates** - More detailed execution progress
- **Request Tracing** - Better request/response correlation

## 🐳 Docker & Deployment

### Container Optimizations
- **Multi-stage Builds** - Reduced container size
- **Health Check Improvements** - Better container health monitoring
- **Volume Management** - Persistent configuration and log storage

### Development Experience
- **Hot Reload Support** - Faster development iteration
- **Debug Logging** - Enhanced debugging capabilities
- **Environment Isolation** - Better separation between services

## 📚 Documentation Updates

### New Documentation
- **Purpose Endpoints Guide** - Comprehensive guide to creating and using purpose endpoints
- **MCP Server Integration** - Updated MCP server setup instructions
- **Windows Installation Guide** - Platform-specific setup documentation

### API Documentation
- **OpenAPI Spec Updates** - Complete API documentation refresh
- **Example Requests** - Expanded request/response examples
- **Error Code Reference** - Comprehensive error handling guide

## 🛠️ Configuration Files

### New Configuration Files
- `config/purpose_endpoints.json` - Purpose endpoint definitions
- Enhanced `config/mcp_configs.json` - Extended MCP server configurations

### Example Configurations
```json
{
  "youtube-video-to-blog": {
    "slug": "youtube-video-to-blog",
    "name": "YouTube Video to Blog",
    "description": "Converts a YouTube video to a blog",
    "prompt_template": "Get the transcript from the following video then write a yakshaver blog post",
    "mcp_servers": ["filesystem-yakshaver-blogs", "youtube-transcript"],
    "max_iterations": 10,
    "timeout": 300
  }
}
```

## 🚨 Breaking Changes

### Configuration Format Changes
- **MCP Configuration Schema** - Enhanced schema with additional fields
- **Environment Variable Names** - Some environment variables renamed for consistency

### API Response Changes
- **Error Response Format** - Standardized error response structure
- **Status Update Schema** - Enhanced status update information

## 📦 Dependencies

### New Dependencies
- **uvx** - Python package manager for MCP servers
- **Enhanced YouTube Support** - YouTube transcript processing libraries

### Updated Dependencies
- **FastAPI** - Updated to latest stable version
- **Pydantic** - Enhanced data validation capabilities
- **Docker Base Images** - Updated to latest security patches

## 🔧 Migration Guide

### Upgrading from v1.0.0

1. **Update Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install uvx (Windows)**
   ```bash
   winget install astral-sh.uv
   ```

3. **Update Configuration Files**
   - Backup existing `config/mcp_configs.json`
   - Run the application to generate new configuration structure
   - Migrate custom configurations

4. **Environment Variables**
   - Review and update environment variable names
   - Add any new required environment variables

## 🎯 Use Cases

### Content Creation Workflow
```bash
# Convert YouTube video to blog post
curl -X POST http://localhost:8000/purpose/youtube-video-to-blog \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=example"}'
```

### Development Automation
```bash
# Analyze code changes
curl -X POST http://localhost:8000/purpose/analyze-code-changes \
  -H "Content-Type: application/json" \
  -d '{"repository": "owner/repo", "commit_sha": "abc123"}'
```

### Research Automation
```bash
# Research a topic comprehensively
curl -X POST http://localhost:8000/purpose/research-topic \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI agent orchestration patterns", "depth": "comprehensive"}'
```

## 🙏 Acknowledgments

- **Community Contributors** - Thanks to all who reported issues and suggested improvements
- **MCP Server Developers** - Gratitude to the creators of the MCP servers we integrate
- **uvx Project** - For providing excellent Python package management

## 📋 Next Steps

### Planned for v1.2.0
- **Custom MCP Server Templates** - Easier creation of custom MCP servers
- **Workflow Scheduling** - Cron-like scheduling for purpose endpoints
- **Multi-language Support** - Support for non-English content processing
- **Enhanced Analytics** - Detailed execution metrics and optimization suggestions

---

**Full Changelog**: [v1.0.0...v1.1.0](https://github.com/calumjs/teloscript/compare/v1.0.0...v1.1.0)

**Download**: [Latest Release](https://github.com/calumjs/teloscript/releases/tag/v1.1.0)

---

**TELOSCRIPT v1.1.0** - *Where purpose meets autonomous coordination* 