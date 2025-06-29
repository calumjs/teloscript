# 🎯 TELOSCRIPT Purpose Endpoints Implementation Plan

## Overview

Purpose Endpoints are predefined, reusable agent configurations that combine prompts, MCP servers, and execution parameters into single endpoints. This allows for rapid deployment of specialized AI workflows without repeated configuration.

## Example Use Case

```
POST /purpose/handle-github-webhook
{
  "webhook_data": {...}
}
```

Automatically uses:
- **Prompt**: "Consume this webhook and process as appropriate"
- **MCP Servers**: ["github-mcp-server"]
- **Configuration**: Predefined parameters for webhook processing

---

## 📋 Implementation Steps

### Phase 1: Data Model Extensions

#### Step 1.1: Extend Models (`src/models.py`)

**Add new Pydantic models:**

```python
class PurposeEndpointConfig(BaseModel):
    """Configuration for a purpose endpoint"""
    slug: str = Field(description="Unique identifier for the endpoint (e.g., 'handle-github-webhook')")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Purpose and functionality description")
    prompt_template: str = Field(description="Template prompt that will be used for all requests")
    mcp_servers: List[str] = Field(description="List of MCP server names to use")
    max_iterations: int = Field(default=10, description="Maximum iterations for goal completion")
    timeout: int = Field(default=300, description="Overall timeout in seconds")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema for input validation")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema for output validation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    enabled: bool = Field(default=True, description="Whether the endpoint is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PurposeEndpointRequest(BaseModel):
    """Request to execute a purpose endpoint"""
    endpoint_slug: str = Field(description="The slug of the purpose endpoint to execute")
    input_data: Dict[str, Any] = Field(description="Input data for the endpoint")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class PurposeEndpointResponse(BaseModel):
    """Response from a purpose endpoint execution"""
    request_id: str
    endpoint_slug: str
    status: AgentStatus
    result: str
    execution_time: float
    iterations_used: int
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: datetime = Field(default_factory=datetime.utcnow)
```

#### Step 1.2: Create Purpose Endpoint Manager (`src/purpose_manager.py`)

**New file for managing purpose endpoints:**

```python
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

from .models import PurposeEndpointConfig, PurposeEndpointRequest, PurposeEndpointResponse
from .mcp_agent import LLMPoweredMCPAgent
from .orchestrator import WorkflowOrchestrator

class PurposeEndpointManager:
    """Manages purpose endpoint configurations and execution"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.endpoints_file = self.config_dir / "purpose_endpoints.json"
        self.endpoints: Dict[str, PurposeEndpointConfig] = {}
        self.orchestrator = WorkflowOrchestrator()
        self.load_endpoints()
    
    def load_endpoints(self) -> None:
        """Load purpose endpoints from configuration file"""
        try:
            self.config_dir.mkdir(exist_ok=True)
            
            if self.endpoints_file.exists():
                with open(self.endpoints_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.endpoints = {
                        slug: PurposeEndpointConfig(**config)
                        for slug, config in data.items()
                    }
                logger.info(f"Loaded {len(self.endpoints)} purpose endpoints")
            else:
                self._create_default_endpoints()
                self.save_endpoints()
                
        except Exception as e:
            logger.error(f"Error loading purpose endpoints: {e}")
            self.endpoints = {}
    
    def save_endpoints(self) -> bool:
        """Save purpose endpoints to configuration file"""
        try:
            data = {
                slug: endpoint.dict()
                for slug, endpoint in self.endpoints.items()
            }
            
            with open(self.endpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.endpoints)} purpose endpoints")
            return True
            
        except Exception as e:
            logger.error(f"Error saving purpose endpoints: {e}")
            return False
    
    def _create_default_endpoints(self) -> None:
        """Create default purpose endpoints"""
        default_endpoints = {
            "handle-github-webhook": PurposeEndpointConfig(
                slug="handle-github-webhook",
                name="GitHub Webhook Handler",
                description="Consume GitHub webhooks and process them appropriately",
                prompt_template="Consume this webhook data and process it as appropriate. Analyze the webhook type, extract relevant information, and take appropriate actions based on the event type and payload.",
                mcp_servers=["github"],
                max_iterations=15,
                timeout=300,
                input_schema={
                    "type": "object",
                    "properties": {
                        "webhook_data": {"type": "object"},
                        "event_type": {"type": "string"},
                        "repository": {"type": "string"}
                    },
                    "required": ["webhook_data"]
                },
                tags=["github", "webhook", "automation"]
            ),
            "analyze-code-changes": PurposeEndpointConfig(
                slug="analyze-code-changes",
                name="Code Change Analyzer",
                description="Analyze code changes and provide insights",
                prompt_template="Analyze the provided code changes. Identify the type of changes, potential impacts, code quality issues, and provide recommendations for improvement.",
                mcp_servers=["filesystem", "github"],
                max_iterations=12,
                timeout=240,
                tags=["code-analysis", "review", "quality"]
            ),
            "research-topic": PurposeEndpointConfig(
                slug="research-topic",
                name="Topic Researcher",
                description="Research a given topic using web search and document findings",
                prompt_template="Research the provided topic thoroughly. Search for current information, gather relevant data, and compile a comprehensive report with sources and insights.",
                mcp_servers=["brave-search", "filesystem"],
                max_iterations=20,
                timeout=600,
                tags=["research", "information-gathering", "analysis"]
            )
        }
        
        self.endpoints.update(default_endpoints)
        logger.info("Created default purpose endpoints")
    
    def get_endpoint(self, slug: str) -> Optional[PurposeEndpointConfig]:
        """Get a purpose endpoint by slug"""
        return self.endpoints.get(slug)
    
    def list_endpoints(self) -> List[PurposeEndpointConfig]:
        """List all purpose endpoints"""
        return list(self.endpoints.values())
    
    def create_endpoint(self, config: PurposeEndpointConfig) -> bool:
        """Create a new purpose endpoint"""
        try:
            if config.slug in self.endpoints:
                raise ValueError(f"Endpoint with slug '{config.slug}' already exists")
            
            self.endpoints[config.slug] = config
            self.save_endpoints()
            logger.info(f"Created purpose endpoint: {config.slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating purpose endpoint: {e}")
            return False
    
    def update_endpoint(self, slug: str, config: PurposeEndpointConfig) -> bool:
        """Update an existing purpose endpoint"""
        try:
            if slug not in self.endpoints:
                raise ValueError(f"Endpoint with slug '{slug}' not found")
            
            config.slug = slug  # Ensure slug matches
            config.updated_at = datetime.utcnow()
            self.endpoints[slug] = config
            self.save_endpoints()
            logger.info(f"Updated purpose endpoint: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating purpose endpoint: {e}")
            return False
    
    def delete_endpoint(self, slug: str) -> bool:
        """Delete a purpose endpoint"""
        try:
            if slug not in self.endpoints:
                raise ValueError(f"Endpoint with slug '{slug}' not found")
            
            del self.endpoints[slug]
            self.save_endpoints()
            logger.info(f"Deleted purpose endpoint: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting purpose endpoint: {e}")
            return False
    
    async def execute_endpoint(self, request: PurposeEndpointRequest, status_callback=None) -> PurposeEndpointResponse:
        """Execute a purpose endpoint"""
        try:
            # Get the endpoint configuration
            endpoint = self.get_endpoint(request.endpoint_slug)
            if not endpoint:
                raise ValueError(f"Purpose endpoint '{request.endpoint_slug}' not found")
            
            if not endpoint.enabled:
                raise ValueError(f"Purpose endpoint '{request.endpoint_slug}' is disabled")
            
            # Validate input if schema is provided
            if endpoint.input_schema:
                # TODO: Add JSON schema validation
                pass
            
            # Build the prompt with input data
            prompt = self._build_prompt(endpoint, request.input_data)
            
            # Create MCP server configurations
            mcp_configs = self._get_mcp_configs(endpoint.mcp_servers)
            
            # Create agent configuration
            from .models import MCPAgentConfig, AgentGoal, MCPServerConfig
            
            agent_config = MCPAgentConfig(
                servers=mcp_configs,
                goal=AgentGoal(
                    description=prompt,
                    context=request.input_data
                ),
                max_iterations=endpoint.max_iterations,
                timeout=endpoint.timeout
            )
            
            # Create agent request
            from .models import AgentRequest
            agent_request = AgentRequest(
                id=request.request_id,
                config=agent_config
            )
            
            # Execute the agent
            start_time = time.time()
            agent_response = await self.orchestrator.run_single_agent(
                agent_request, 
                status_callback
            )
            
            execution_time = time.time() - start_time
            
            # Build response
            response = PurposeEndpointResponse(
                request_id=request.request_id,
                endpoint_slug=request.endpoint_slug,
                status=agent_response.status,
                result=agent_response.result,
                execution_time=execution_time,
                iterations_used=agent_response.iterations_used,
                input_data=request.input_data,
                error=agent_response.error,
                completed_at=datetime.utcnow()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing purpose endpoint: {e}")
            return PurposeEndpointResponse(
                request_id=request.request_id,
                endpoint_slug=request.endpoint_slug,
                status=AgentStatus.FAILED,
                result="",
                execution_time=0,
                iterations_used=0,
                input_data=request.input_data,
                error=str(e),
                completed_at=datetime.utcnow()
            )
    
    def _build_prompt(self, endpoint: PurposeEndpointConfig, input_data: Dict[str, Any]) -> str:
        """Build the final prompt by combining template with input data"""
        # Simple template substitution for now
        # Could be enhanced with more sophisticated templating
        prompt = endpoint.prompt_template
        
        # Add input data context
        if input_data:
            prompt += f"\n\nInput Data:\n{json.dumps(input_data, indent=2)}"
        
        return prompt
    
    def _get_mcp_configs(self, server_names: List[str]) -> List[MCPServerConfig]:
        """Get MCP server configurations by name"""
        # Load MCP configs from the existing system
        from .api import load_mcp_configs
        mcp_configs = load_mcp_configs()
        
        server_configs = []
        for server_name in server_names:
            if server_name in mcp_configs:
                config_data = mcp_configs[server_name]["config"]
                server_configs.append(MCPServerConfig(**config_data))
            else:
                logger.warning(f"MCP server '{server_name}' not found in configuration")
        
        return server_configs
```

### Phase 2: API Integration

#### Step 2.1: Extend API (`src/api.py`)

**Add new imports and initialization:**

```python
# Add to existing imports
from .purpose_manager import PurposeEndpointManager

# Add to global variables
purpose_manager = PurposeEndpointManager()
```

**Add new API endpoints:**

```python
# Purpose Endpoint Management
@app.get("/purpose/endpoints")
async def list_purpose_endpoints():
    """List all available purpose endpoints"""
    endpoints = purpose_manager.list_endpoints()
    return {
        "endpoints": [endpoint.dict() for endpoint in endpoints],
        "count": len(endpoints)
    }

@app.get("/purpose/endpoints/{slug}")
async def get_purpose_endpoint(slug: str):
    """Get a specific purpose endpoint"""
    endpoint = purpose_manager.get_endpoint(slug)
    if not endpoint:
        raise HTTPException(status_code=404, detail=f"Purpose endpoint '{slug}' not found")
    return endpoint.dict()

@app.post("/purpose/endpoints")
async def create_purpose_endpoint(request: Request):
    """Create a new purpose endpoint"""
    try:
        data = await request.json()
        config = PurposeEndpointConfig(**data)
        
        if purpose_manager.create_endpoint(config):
            return {"message": f"Purpose endpoint '{config.slug}' created successfully", "endpoint": config.dict()}
        else:
            raise HTTPException(status_code=500, detail="Failed to create purpose endpoint")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/purpose/endpoints/{slug}")
async def update_purpose_endpoint(slug: str, request: Request):
    """Update an existing purpose endpoint"""
    try:
        data = await request.json()
        config = PurposeEndpointConfig(**data)
        
        if purpose_manager.update_endpoint(slug, config):
            return {"message": f"Purpose endpoint '{slug}' updated successfully", "endpoint": config.dict()}
        else:
            raise HTTPException(status_code=500, detail="Failed to update purpose endpoint")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/purpose/endpoints/{slug}")
async def delete_purpose_endpoint(slug: str):
    """Delete a purpose endpoint"""
    if purpose_manager.delete_endpoint(slug):
        return {"message": f"Purpose endpoint '{slug}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Purpose endpoint '{slug}' not found")

# Purpose Endpoint Execution
@app.post("/purpose/{slug}")
async def execute_purpose_endpoint(slug: str, request: Request):
    """Execute a purpose endpoint by slug"""
    try:
        # Get input data
        input_data = await request.json()
        
        # Create purpose endpoint request
        purpose_request = PurposeEndpointRequest(
            endpoint_slug=slug,
            input_data=input_data
        )
        
        # Execute the endpoint
        response = await purpose_manager.execute_endpoint(purpose_request)
        
        return response.dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/purpose/{slug}/stream")
async def execute_purpose_endpoint_stream(slug: str, request: Request):
    """Execute a purpose endpoint with streaming updates"""
    try:
        # Get input data
        input_data = await request.json()
        
        # Create purpose endpoint request
        purpose_request = PurposeEndpointRequest(
            endpoint_slug=slug,
            input_data=input_data
        )
        
        # Create status callback for streaming
        status_updates = []
        
        async def stream_callback(update: StatusUpdate):
            status_updates.append(update)
            # In a real implementation, you'd stream this to the client
        
        # Execute the endpoint
        response = await purpose_manager.execute_endpoint(purpose_request, stream_callback)
        
        return {
            "response": response.dict(),
            "status_updates": [update.dict() for update in status_updates]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Phase 3: Configuration Management

#### Step 3.1: Create Default Configuration (`config/purpose_endpoints.json`)

```json
{
  "handle-github-webhook": {
    "slug": "handle-github-webhook",
    "name": "GitHub Webhook Handler",
    "description": "Consume GitHub webhooks and process them appropriately",
    "prompt_template": "Consume this webhook data and process it as appropriate. Analyze the webhook type, extract relevant information, and take appropriate actions based on the event type and payload. Consider the repository context, user permissions, and event-specific requirements.",
    "mcp_servers": ["github", "filesystem"],
    "max_iterations": 15,
    "timeout": 300,
    "input_schema": {
      "type": "object",
      "properties": {
        "webhook_data": {
          "type": "object",
          "description": "The complete webhook payload from GitHub"
        },
        "event_type": {
          "type": "string",
          "description": "The type of GitHub event (e.g., push, pull_request, issues)"
        },
        "repository": {
          "type": "string",
          "description": "Repository name in format owner/repo"
        },
        "action": {
          "type": "string",
          "description": "The specific action within the event type"
        }
      },
      "required": ["webhook_data"]
    },
    "tags": ["github", "webhook", "automation", "integration"],
    "enabled": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "analyze-code-changes": {
    "slug": "analyze-code-changes",
    "name": "Code Change Analyzer",
    "description": "Analyze code changes and provide insights",
    "prompt_template": "Analyze the provided code changes thoroughly. Identify the type of changes, potential impacts on the codebase, code quality issues, security concerns, and provide specific recommendations for improvement. Consider best practices, maintainability, and potential risks.",
    "mcp_servers": ["filesystem", "github"],
    "max_iterations": 12,
    "timeout": 240,
    "input_schema": {
      "type": "object",
      "properties": {
        "file_paths": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of file paths to analyze"
        },
        "diff_data": {
          "type": "object",
          "description": "Git diff information"
        },
        "commit_message": {
          "type": "string",
          "description": "Commit message for context"
        }
      },
      "required": ["file_paths"]
    },
    "tags": ["code-analysis", "review", "quality", "development"],
    "enabled": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "research-topic": {
    "slug": "research-topic",
    "name": "Topic Researcher",
    "description": "Research a given topic using web search and document findings",
    "prompt_template": "Research the provided topic thoroughly using available resources. Search for current information, gather relevant data from multiple sources, analyze trends and patterns, and compile a comprehensive report with proper citations, insights, and actionable conclusions.",
    "mcp_servers": ["brave-search", "filesystem", "tavily"],
    "max_iterations": 20,
    "timeout": 600,
    "input_schema": {
      "type": "object",
      "properties": {
        "topic": {
          "type": "string",
          "description": "The topic to research"
        },
        "scope": {
          "type": "string",
          "description": "Research scope and focus areas"
        },
        "output_format": {
          "type": "string",
          "enum": ["report", "summary", "detailed"],
          "description": "Desired output format"
        }
      },
      "required": ["topic"]
    },
    "tags": ["research", "information-gathering", "analysis", "documentation"],
    "enabled": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### Phase 4: Web Interface Integration

#### Step 4.1: Add Purpose Endpoints to Dashboard

**Extend the web interface in `src/api.py`:**

```python
# Add to the orchestration interface HTML
# In the nav-tabs section, add:
"""
<div class="nav-tab" onclick="showTab('purpose-endpoints')">
    🎯 Purpose Endpoints
</div>
"""

# Add new tab content:
"""
<div id="purpose-endpoints" class="tab-content" style="display: none;">
    <div class="purpose-endpoints-container">
        <div class="purpose-endpoints-header">
            <h2>🎯 Purpose Endpoints</h2>
            <p>Predefined agent configurations for common tasks</p>
        </div>
        
        <div class="purpose-endpoints-grid" id="purpose-endpoints-grid">
            <!-- Dynamic content -->
        </div>
        
        <div class="purpose-endpoint-form" id="purpose-endpoint-form" style="display: none;">
            <h3>Create New Purpose Endpoint</h3>
            <form id="new-purpose-endpoint-form">
                <div class="form-group">
                    <label>Slug:</label>
                    <input type="text" id="endpoint-slug" placeholder="e.g., handle-github-webhook" required>
                </div>
                <div class="form-group">
                    <label>Name:</label>
                    <input type="text" id="endpoint-name" placeholder="Human-readable name" required>
                </div>
                <div class="form-group">
                    <label>Description:</label>
                    <textarea id="endpoint-description" placeholder="Purpose and functionality description" required></textarea>
                </div>
                <div class="form-group">
                    <label>Prompt Template:</label>
                    <textarea id="endpoint-prompt" placeholder="Template prompt for the endpoint" required></textarea>
                </div>
                <div class="form-group">
                    <label>MCP Servers:</label>
                    <select id="endpoint-servers" multiple>
                        <!-- Dynamic server options -->
                    </select>
                </div>
                <div class="form-group">
                    <label>Max Iterations:</label>
                    <input type="number" id="endpoint-iterations" value="10" min="1" max="50">
                </div>
                <div class="form-group">
                    <label>Timeout (seconds):</label>
                    <input type="number" id="endpoint-timeout" value="300" min="30" max="3600">
                </div>
                <div class="form-actions">
                    <button type="submit">Create Endpoint</button>
                    <button type="button" onclick="hidePurposeEndpointForm()">Cancel</button>
                </div>
            </form>
        </div>
        
        <div class="purpose-endpoints-actions">
            <button onclick="showPurposeEndpointForm()" class="btn-primary">
                ➕ Create New Endpoint
            </button>
            <button onclick="refreshPurposeEndpoints()" class="btn-secondary">
                🔄 Refresh
            </button>
        </div>
    </div>
</div>
"""

# Add JavaScript functions:
"""
<script>
// Purpose Endpoints Management
async function loadPurposeEndpoints() {
    try {
        const response = await fetch('/purpose/endpoints');
        const data = await response.json();
        
        const grid = document.getElementById('purpose-endpoints-grid');
        grid.innerHTML = '';
        
        data.endpoints.forEach(endpoint => {
            const card = createPurposeEndpointCard(endpoint);
            grid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading purpose endpoints:', error);
    }
}

function createPurposeEndpointCard(endpoint) {
    const card = document.createElement('div');
    card.className = 'purpose-endpoint-card';
    card.innerHTML = `
        <div class="card-header">
            <h3>${endpoint.name}</h3>
            <span class="slug">/${endpoint.slug}</span>
        </div>
        <div class="card-body">
            <p>${endpoint.description}</p>
            <div class="endpoint-meta">
                <span class="servers">🖥️ ${endpoint.mcp_servers.join(', ')}</span>
                <span class="status ${endpoint.enabled ? 'enabled' : 'disabled'}">
                    ${endpoint.enabled ? '✅ Enabled' : '❌ Disabled'}
                </span>
            </div>
        </div>
        <div class="card-actions">
            <button onclick="testPurposeEndpoint('${endpoint.slug}')" class="btn-test">
                🚀 Test
            </button>
            <button onclick="editPurposeEndpoint('${endpoint.slug}')" class="btn-edit">
                ✏️ Edit
            </button>
            <button onclick="deletePurposeEndpoint('${endpoint.slug}')" class="btn-delete">
                🗑️ Delete
            </button>
        </div>
    `;
    return card;
}

async function testPurposeEndpoint(slug) {
    const testData = prompt(`Enter test data for ${slug} (JSON format):`, '{}');
    if (!testData) return;
    
    try {
        const inputData = JSON.parse(testData);
        const response = await fetch(`/purpose/${slug}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        
        const result = await response.json();
        alert(`Result: ${JSON.stringify(result, null, 2)}`);
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Add to existing tab switching function
function showTab(tabName) {
    // ... existing code ...
    
    if (tabName === 'purpose-endpoints') {
        loadPurposeEndpoints();
    }
}
</script>
"""
```

### Phase 5: Testing and Validation

#### Step 5.1: Create Test Scripts

**Create `tests/test_purpose_endpoints.py`:**

```python
import pytest
import asyncio
from src.purpose_manager import PurposeEndpointManager
from src.models import PurposeEndpointConfig, PurposeEndpointRequest

@pytest.fixture
def purpose_manager():
    return PurposeEndpointManager("test_config")

@pytest.mark.asyncio
async def test_create_purpose_endpoint(purpose_manager):
    config = PurposeEndpointConfig(
        slug="test-endpoint",
        name="Test Endpoint",
        description="A test purpose endpoint",
        prompt_template="Process the input data",
        mcp_servers=["filesystem"]
    )
    
    assert purpose_manager.create_endpoint(config) == True
    assert purpose_manager.get_endpoint("test-endpoint") is not None

@pytest.mark.asyncio
async def test_execute_purpose_endpoint(purpose_manager):
    # Create test endpoint
    config = PurposeEndpointConfig(
        slug="test-execution",
        name="Test Execution",
        description="Test execution endpoint",
        prompt_template="Analyze this data: {input_data}",
        mcp_servers=["filesystem"]
    )
    purpose_manager.create_endpoint(config)
    
    # Execute endpoint
    request = PurposeEndpointRequest(
        endpoint_slug="test-execution",
        input_data={"test": "data"}
    )
    
    response = await purpose_manager.execute_endpoint(request)
    assert response.endpoint_slug == "test-execution"
    assert response.status in ["completed", "failed"]
```

#### Step 5.2: Integration Testing

**Create `tests/test_api_integration.py`:**

```python
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_list_purpose_endpoints():
    response = client.get("/purpose/endpoints")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "count" in data

def test_create_purpose_endpoint():
    endpoint_data = {
        "slug": "test-api-endpoint",
        "name": "Test API Endpoint",
        "description": "Test endpoint via API",
        "prompt_template": "Process this data",
        "mcp_servers": ["filesystem"]
    }
    
    response = client.post("/purpose/endpoints", json=endpoint_data)
    assert response.status_code == 200

def test_execute_purpose_endpoint():
    # First create an endpoint
    endpoint_data = {
        "slug": "test-execution-api",
        "name": "Test Execution API",
        "description": "Test execution via API",
        "prompt_template": "Analyze: {input_data}",
        "mcp_servers": ["filesystem"]
    }
    client.post("/purpose/endpoints", json=endpoint_data)
    
    # Then execute it
    input_data = {"test": "execution data"}
    response = client.post("/purpose/test-execution-api", json=input_data)
    assert response.status_code == 200
```

### Phase 6: Documentation and Examples

#### Step 6.1: Update README

**Add to `README.md`:**

```markdown
## 🎯 Purpose Endpoints

Purpose Endpoints are predefined, reusable agent configurations that combine prompts, MCP servers, and execution parameters into single endpoints.

### Quick Start

```bash
# List available purpose endpoints
curl http://localhost:8000/purpose/endpoints

# Execute a purpose endpoint
curl -X POST http://localhost:8000/purpose/handle-github-webhook \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_data": {
      "event_type": "push",
      "repository": "owner/repo",
      "commits": [...]
    }
  }'
```

### Creating Custom Endpoints

```bash
curl -X POST http://localhost:8000/purpose/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "my-custom-endpoint",
    "name": "My Custom Endpoint",
    "description": "Process custom data",
    "prompt_template": "Analyze this data: {input_data}",
    "mcp_servers": ["filesystem", "brave-search"],
    "max_iterations": 10,
    "timeout": 300
  }'
```

### Available Endpoints

- **handle-github-webhook**: Process GitHub webhooks automatically
- **analyze-code-changes**: Analyze code changes and provide insights
- **research-topic**: Research topics using web search and document findings
```

#### Step 6.2: Create Examples Directory

**Create `examples/purpose_endpoints/`:**

```bash
mkdir -p examples/purpose_endpoints
```

**Create example files:**

```json
// examples/purpose_endpoints/github-webhook-example.json
{
  "webhook_data": {
    "event_type": "push",
    "repository": "calumjs/teloscript",
    "commits": [
      {
        "id": "abc123",
        "message": "Add new feature",
        "author": "calumjs"
      }
    ]
  }
}
```

```json
// examples/purpose_endpoints/code-analysis-example.json
{
  "file_paths": ["src/api.py", "src/mcp_agent.py"],
  "diff_data": {
    "files_changed": 2,
    "lines_added": 50,
    "lines_deleted": 10
  },
  "commit_message": "Add purpose endpoints feature"
}
```

### Phase 7: Deployment and Configuration

#### Step 7.1: Update Docker Configuration

**Update `docker-compose.yml`:**

```yaml
services:
  teloscript-api:
    # ... existing configuration ...
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
      - ./examples:/app/examples  # Add examples directory
```

#### Step 7.2: Update Startup Script

**Update `scripts/startup.sh`:**

```bash
# Add to the main() function
echo "🎯 Loading Purpose Endpoints..."
if [[ -f "config/purpose_endpoints.json" ]]; then
    echo "  ✅ Found purpose endpoints configuration"
    endpoint_count=$(jq '. | length' config/purpose_endpoints.json 2>/dev/null || echo "0")
    echo "  📊 Loaded $endpoint_count purpose endpoints"
else
    echo "  ⚠️  No purpose endpoints configuration found"
fi
```

---

## 🚀 Implementation Timeline

### Week 1: Foundation
- [ ] Phase 1: Data Model Extensions
- [ ] Phase 2: Basic API Integration
- [ ] Unit tests for core functionality

### Week 2: Core Features
- [ ] Phase 3: Configuration Management
- [ ] Phase 4: Web Interface Integration
- [ ] Integration testing

### Week 3: Polish and Documentation
- [ ] Phase 5: Testing and Validation
- [ ] Phase 6: Documentation and Examples
- [ ] Phase 7: Deployment and Configuration

### Week 4: Testing and Deployment
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Production deployment

---

## 🎯 Success Criteria

1. **Functional Requirements**
   - ✅ Create purpose endpoints with baked-in prompts and MCP servers
   - ✅ Execute purpose endpoints via REST API
   - ✅ Manage purpose endpoints (CRUD operations)
   - ✅ Web interface for purpose endpoint management

2. **Technical Requirements**
   - ✅ Proper error handling and validation
   - ✅ Configuration persistence
   - ✅ Real-time status updates
   - ✅ Comprehensive logging

3. **User Experience**
   - ✅ Intuitive web interface
   - ✅ Clear documentation and examples
   - ✅ Easy endpoint creation and management
   - ✅ Fast execution and response times

4. **Production Readiness**
   - ✅ Docker integration
   - ✅ Health checks and monitoring
   - ✅ Security considerations
   - ✅ Scalability considerations

---

## 🔧 Future Enhancements

1. **Advanced Templating**: Support for more sophisticated prompt templating with variables, conditionals, and loops
2. **Endpoint Chaining**: Ability to chain purpose endpoints together
3. **Input/Output Validation**: JSON schema validation for inputs and outputs
4. **Rate Limiting**: Per-endpoint rate limiting and quotas
5. **Authentication**: Endpoint-level authentication and authorization
6. **Monitoring**: Detailed metrics and analytics for endpoint usage
7. **Versioning**: Support for endpoint versioning and rollbacks
8. **Templates**: Pre-built endpoint templates for common use cases

This implementation plan provides a comprehensive roadmap for adding Purpose Endpoints to TELOSCRIPT, creating a powerful and flexible system for predefined AI workflows. 